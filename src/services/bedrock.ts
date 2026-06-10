import {
  BedrockAgentRuntimeClient,
  RetrieveCommand,
  RetrieveAndGenerateCommand,
  type RetrieveCommandInput,
  type RetrieveAndGenerateCommandInput,
  type RetrievalResultLocation,
  type RetrievalFilter,
} from "@aws-sdk/client-bedrock-agent-runtime";
import { fromIni } from "@aws-sdk/credential-providers";
import { getCognitoCredentials, CognitoAuthRequiredError } from "../auth/cognito.js";

// ── Configuration ────────────────────────────────────────────────────────────

const AWS_REGION = process.env.AWS_REGION ?? "us-east-1";
const AWS_PROFILE = process.env.AWS_PROFILE ?? "default";

// ── Client (singleton) ───────────────────────────────────────────────────────

let _client: BedrockAgentRuntimeClient | null = null;

export async function getBedrockClient(): Promise<BedrockAgentRuntimeClient> {
  if (!_client) {
    if (process.env.COGNITO_IDENTITY_POOL_ID) {
      try {
        const credentials = await getCognitoCredentials(AWS_REGION);
        _client = new BedrockAgentRuntimeClient({ region: AWS_REGION, credentials });
      } catch (err) {
        if (err instanceof CognitoAuthRequiredError) throw err;
        throw err;
      }
    } else {
      _client = new BedrockAgentRuntimeClient({
        region: AWS_REGION,
        credentials: fromIni({ profile: AWS_PROFILE }),
      });
    }
  }
  return _client;
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface RetrievedPassage {
  content: string;
  score: number | undefined;
  location: RetrievalResultLocation | undefined;
  metadata: Record<string, unknown> | undefined;
}

export interface RetrieveResult {
  passages: RetrievedPassage[];
  nextToken: string | undefined;
}

export interface RetrieveAndGenerateResult {
  output: string;
  sessionId: string | undefined;
  citations: Citation[];
}

export interface Citation {
  generatedText: string | undefined;
  passages: RetrievedPassage[];
}

// ── Retrieve ─────────────────────────────────────────────────────────────────

export interface RetrieveParams {
  knowledgeBaseId: string;
  query: string;
  numberOfResults?: number;
  searchType?: "SEMANTIC" | "HYBRID";
  nextToken?: string;
  metadataFilter?: Record<string, unknown>;
}

export async function retrieve(params: RetrieveParams): Promise<RetrieveResult> {
  const client = await getBedrockClient();

  const input: RetrieveCommandInput = {
    knowledgeBaseId: params.knowledgeBaseId,
    retrievalQuery: { text: params.query },
    retrievalConfiguration: {
      vectorSearchConfiguration: {
        numberOfResults: params.numberOfResults ?? 5,
        ...(params.searchType ? { overrideSearchType: params.searchType } : {}),
        ...(params.metadataFilter
          ? { filter: params.metadataFilter as unknown as RetrievalFilter }
          : {}),
      },
    },
    ...(params.nextToken ? { nextToken: params.nextToken } : {}),
  };

  const command = new RetrieveCommand(input);
  const response = await client.send(command);

  const passages: RetrievedPassage[] = (response.retrievalResults ?? []).map((r) => ({
    content: r.content?.text ?? "",
    score: r.score,
    location: r.location,
    metadata: r.metadata as Record<string, unknown> | undefined,
  }));

  return {
    passages,
    nextToken: response.nextToken,
  };
}

// ── Retrieve and Generate ────────────────────────────────────────────────────

export interface RetrieveAndGenerateParams {
  knowledgeBaseId: string;
  modelArn: string;
  query: string;
  numberOfResults?: number;
  searchType?: "SEMANTIC" | "HYBRID";
  sessionId?: string;
  systemPrompt?: string;
}

export async function retrieveAndGenerate(
  params: RetrieveAndGenerateParams
): Promise<RetrieveAndGenerateResult> {
  const client = await getBedrockClient();

  const input: RetrieveAndGenerateCommandInput = {
    input: { text: params.query },
    retrieveAndGenerateConfiguration: {
      type: "KNOWLEDGE_BASE",
      knowledgeBaseConfiguration: {
        knowledgeBaseId: params.knowledgeBaseId,
        modelArn: params.modelArn,
        retrievalConfiguration: {
          vectorSearchConfiguration: {
            numberOfResults: params.numberOfResults ?? 5,
            ...(params.searchType ? { overrideSearchType: params.searchType } : {}),
          },
        },
        ...(params.systemPrompt
          ? {
              generationConfiguration: {
                promptTemplate: {
                  textPromptTemplate: params.systemPrompt,
                },
              },
            }
          : {}),
      },
    },
    ...(params.sessionId ? { sessionId: params.sessionId } : {}),
  };

  const command = new RetrieveAndGenerateCommand(input);
  const response = await client.send(command);

  const citations: Citation[] = (response.citations ?? []).map((c) => ({
    generatedText: c.generatedResponsePart?.textResponsePart?.text,
    passages: (c.retrievedReferences ?? []).map((ref) => ({
      content: ref.content?.text ?? "",
      score: undefined,
      location: ref.location,
      metadata: ref.metadata as Record<string, unknown> | undefined,
    })),
  }));

  return {
    output: response.output?.text ?? "",
    sessionId: response.sessionId,
    citations,
  };
}

// ── Error helper ─────────────────────────────────────────────────────────────

export function formatAwsError(error: unknown): string {
  if (error instanceof Error) {
    const name = error.name;
    const msg = error.message;

    if (name === "ResourceNotFoundException") {
      return `Error: Knowledge Base not found. Check that BEDROCK_KNOWLEDGE_BASE_ID is correct and the KB exists in region ${AWS_REGION}.`;
    }
    if (name === "AccessDeniedException") {
      return `Error: Access denied. Ensure the AWS profile '${AWS_PROFILE}' has the 'bedrock:Retrieve' and 'bedrock:RetrieveAndGenerate' IAM permissions.`;
    }
    if (name === "ValidationException") {
      return `Error: Invalid request — ${msg}. Check the parameters and try again.`;
    }
    if (name === "ThrottlingException") {
      return "Error: AWS request throttled. Wait a moment and retry.";
    }
    if (name === "CredentialsProviderError") {
      return process.env.COGNITO_IDENTITY_POOL_ID
        ? `Error: Could not obtain AWS credentials via Cognito. Delete ~/.atlas-ai/cognito-credentials.json and restart to re-authenticate.`
        : `Error: Could not load AWS credentials for profile '${AWS_PROFILE}'. Run 'aws sso login --profile ${AWS_PROFILE}' or check your ~/.aws/credentials file.`;
    }
    return `Error [${name}]: ${msg}`;
  }
  return `Error: ${String(error)}`;
}
