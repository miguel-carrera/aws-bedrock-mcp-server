import { fromCognitoIdentityPool } from "@aws-sdk/credential-provider-cognito-identity";
import { createServer } from "http";
import { readFile, writeFile, mkdir } from "fs/promises";
import { homedir } from "os";
import { join } from "path";
import open from "open";

const CACHE_PATH = join(homedir(), ".atlas-ai", "cognito-credentials.json");
const CALLBACK_PORT = 9876;
const CALLBACK_URL = `http://localhost:${CALLBACK_PORT}/callback`;
const AUTH_TIMEOUT_MS = 5 * 60 * 1000;

// ── Error type ────────────────────────────────────────────────────────────────

export class CognitoAuthRequiredError extends Error {
  constructor(public readonly inProgress: boolean) {
    super(
      inProgress
        ? "AWS login already in progress — please complete the login in your browser, then retry."
        : "AWS login required — a browser window has been opened for you to log in. Once you complete the login, retry your request.",
    );
    this.name = "CognitoAuthRequiredError";
  }
}

// ── In-progress state (module-level, shared across calls in this process) ────

let authInProgress = false;

// ── Cache ─────────────────────────────────────────────────────────────────────

interface CachedCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken: string;
  expiration: string;
}

async function readCache(): Promise<CachedCredentials | null> {
  try {
    const raw = await readFile(CACHE_PATH, "utf8");
    const cached: CachedCredentials = JSON.parse(raw);
    // Keep 60s buffer before expiry
    if (new Date(cached.expiration).getTime() > Date.now() + 60_000) {
      return cached;
    }
  } catch {
    // no cache or parse error
  }
  return null;
}

async function writeCache(creds: CachedCredentials): Promise<void> {
  await mkdir(join(homedir(), ".atlas-ai"), { recursive: true });
  await writeFile(CACHE_PATH, JSON.stringify(creds, null, 2), "utf8");
}

// ── Browser auth flow (runs in background after throwing) ────────────────────

function startCallbackServer(domain: string, clientId: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      server.close();
      reject(new Error("Cognito auth timed out after 5 minutes"));
    }, AUTH_TIMEOUT_MS);

    const server = createServer((req, res) => {
      const url = new URL(req.url ?? "/", `http://localhost:${CALLBACK_PORT}`);
      const code = url.searchParams.get("code");
      const error = url.searchParams.get("error");

      if (code) {
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end("<html><body><h2>Authentication successful — you can close this tab.</h2></body></html>");
        clearTimeout(timeout);
        server.close();
        resolve(code);
      } else {
        res.writeHead(400, { "Content-Type": "text/html" });
        res.end(`<html><body><h2>Authentication failed: ${error ?? "unknown error"}</h2></body></html>`);
        clearTimeout(timeout);
        server.close();
        reject(new Error(`Cognito auth failed: ${error ?? "unknown"}`));
      }
    });

    server.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });

    server.listen(CALLBACK_PORT, () => {
      const loginUrl =
        `https://${domain}/oauth2/authorize` +
        `?response_type=code` +
        `&client_id=${clientId}` +
        `&redirect_uri=${encodeURIComponent(CALLBACK_URL)}` +
        `&scope=openid+email+profile`;

      console.error(`[cognito-auth] Opening browser for login:\n  ${loginUrl}`);
      open(loginUrl);
    });
  });
}

async function exchangeCodeForTokens(
  domain: string,
  clientId: string,
  code: string,
): Promise<{ id_token: string }> {
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    code,
    redirect_uri: CALLBACK_URL,
    client_id: clientId,
  });

  const resp = await fetch(`https://${domain}/oauth2/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Token exchange failed (${resp.status}): ${text}`);
  }

  return resp.json() as Promise<{ id_token: string }>;
}

// Extract the login key for the Identity Pool logins map directly from the
// JWT issuer claim. This avoids any case-sensitivity issues with domain-based
// derivation — the token always has the canonical user pool ID.
function loginKeyFromIdToken(idToken: string): string {
  const payload = JSON.parse(
    Buffer.from(idToken.split(".")[1], "base64url").toString(),
  ) as { iss: string };
  // iss is "https://cognito-idp.{region}.amazonaws.com/{userPoolId}"
  // Identity Pool expects the same string without the "https://" prefix
  return payload.iss.replace("https://", "");
}

function runBackgroundAuth(domain: string, clientId: string, identityPoolId: string, region: string): void {
  startCallbackServer(domain, clientId)
    .then((code) => exchangeCodeForTokens(domain, clientId, code))
    .then(async (tokens) => {
      const loginKey = loginKeyFromIdToken(tokens.id_token);
      const provider = fromCognitoIdentityPool({
        clientConfig: { region },
        identityPoolId,
        logins: { [loginKey]: tokens.id_token },
      });
      const awsCreds = await provider();
      const expiration =
        awsCreds.expiration?.toISOString() ??
        new Date(Date.now() + 3_600_000).toISOString();
      await writeCache({
        accessKeyId: awsCreds.accessKeyId,
        secretAccessKey: awsCreds.secretAccessKey,
        sessionToken: awsCreds.sessionToken ?? "",
        expiration,
      });
      console.error("[cognito-auth] Credentials obtained and cached. You can now retry your request.");
    })
    .catch((err) => {
      console.error("[cognito-auth] Background auth failed:", err.message);
    })
    .finally(() => {
      authInProgress = false;
    });
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function getCognitoCredentials(region: string = "us-east-1"): Promise<{
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
}> {
  const identityPoolId = process.env.COGNITO_IDENTITY_POOL_ID!;
  const clientId = process.env.COGNITO_USER_POOL_CLIENT_ID!;
  const domain = process.env.COGNITO_DOMAIN!;

  const cached = await readCache();
  if (cached) {
    return {
      accessKeyId: cached.accessKeyId,
      secretAccessKey: cached.secretAccessKey,
      sessionToken: cached.sessionToken,
    };
  }

  if (authInProgress) {
    throw new CognitoAuthRequiredError(true);
  }

  // Start background auth and immediately throw so Claude can inform the user
  authInProgress = true;
  runBackgroundAuth(domain, clientId, identityPoolId, region);
  throw new CognitoAuthRequiredError(false);
}
