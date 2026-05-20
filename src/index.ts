#!/usr/bin/env node
/**
 * MCP Server for AWS Bedrock Knowledge Base Retrieve API.
 *
 * Provides two tools:
 *  - bedrock_retrieve              Semantic / hybrid search over a Knowledge Base
 *  - bedrock_retrieve_and_generate RAG: retrieve + generate an answer via a Bedrock model
 *
 * Configuration (environment variables):
 *  BEDROCK_KNOWLEDGE_BASE_ID  Default Knowledge Base ID (required unless passed per-call)
 *  AWS_REGION                 AWS region (default: us-east-1)
 *  AWS_PROFILE                AWS named profile (default: default)
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import {
  retrieve,
  retrieveAndGenerate,
  formatAwsError,
  type RetrievedPassage,
  type Citation,
} from "./services/bedrock.js";

// ── Constants ─────────────────────────────────────────────────────────────────

const CHARACTER_LIMIT = 25_000;

const DEFAULT_KB_ID = process.env.BEDROCK_KNOWLEDGE_BASE_ID ?? "";

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatLocation(location: RetrievedPassage["location"]): string {
  if (!location) return "unknown";
  if (location.type === "S3") return location.s3Location?.uri ?? "s3://unknown";
  if (location.type === "WEB") return location.webLocation?.url ?? "web://unknown";
  if (location.type === "CONFLUENCE") return location.confluenceLocation?.url ?? "confluence://unknown";
  if (location.type === "SALESFORCE") return location.salesforceLocation?.url ?? "salesforce://unknown";
  if (location.type === "SHAREPOINT") return location.sharePointLocation?.url ?? "sharepoint://unknown";
  return location.type ?? "unknown";
}

function passagesToMarkdown(passages: RetrievedPassage[]): string {
  return passages
    .map((p, i) => {
      const score = p.score !== undefined ? ` (score: ${p.score.toFixed(4)})` : "";
      const src = formatLocation(p.location);
      return `### Result ${i + 1}${score}\n**Source**: ${src}\n\n${p.content}`;
    })
    .join("\n\n---\n\n");
}

function citationsToMarkdown(citations: Citation[]): string {
  return citations
    .map((c, i) => {
      const refs = c.passages
        .map((p) => `  - ${formatLocation(p.location)}`)
        .join("\n");
      return `**[${i + 1}]** ${c.generatedText ?? ""}\n*Sources*:\n${refs}`;
    })
    .join("\n\n");
}

function truncate(text: string): { text: string; truncated: boolean } {
  if (text.length <= CHARACTER_LIMIT) return { text, truncated: false };
  return {
    text: text.slice(0, CHARACTER_LIMIT) + "\n\n[Response truncated. Use `numberOfResults` or a more specific query to reduce output size.]",
    truncated: true,
  };
}

// ── Server ────────────────────────────────────────────────────────────────────

const server = new McpServer({
  name: "bedrock-mcp-server",
  version: "1.0.0",
});

// ── Tool: bedrock_retrieve ────────────────────────────────────────────────────

const RetrieveInputSchema = z
  .object({
    query: z
      .string()
      .min(1, "Query must not be empty")
      .max(1000, "Query must not exceed 1000 characters")
      .describe("Natural language query to search the Knowledge Base"),
    knowledge_base_id: z
      .string()
      .optional()
      .describe(
        "Bedrock Knowledge Base ID. Defaults to BEDROCK_KNOWLEDGE_BASE_ID env var if not provided."
      ),
    number_of_results: z
      .number()
      .int()
      .min(1)
      .max(100)
      .default(5)
      .describe("Number of passages to retrieve (1–100, default: 5)"),
    search_type: z
      .enum(["SEMANTIC", "HYBRID"])
      .default("SEMANTIC")
      .describe(
        "Search strategy: 'SEMANTIC' for pure vector search, 'HYBRID' for vector + keyword (default: SEMANTIC)"
      ),
    next_token: z
      .string()
      .optional()
      .describe("Pagination token from a previous response to fetch the next page of results"),
    response_format: z
      .enum(["markdown", "json"])
      .default("markdown")
      .describe("Output format: 'markdown' for human-readable, 'json' for structured data"),
  })
  .strict();

type RetrieveInput = z.infer<typeof RetrieveInputSchema>;

server.registerTool(
  "bedrock_retrieve",
  {
    title: "Retrieve from Bedrock Knowledge Base",
    description: `Perform a semantic or hybrid search against an AWS Bedrock Knowledge Base and return the most relevant text passages.

Use this tool when you need to find specific information stored in the Knowledge Base — for example, product documentation, FAQs, internal policies, or any indexed content.

Args:
  - query (string): Natural language question or search phrase (required)
  - knowledge_base_id (string): Bedrock KB ID (e.g. "ABCDE12345"). Defaults to BEDROCK_KNOWLEDGE_BASE_ID env var.
  - number_of_results (number): How many passages to return, 1–100 (default: 5)
  - search_type ('SEMANTIC' | 'HYBRID'): Search strategy (default: 'SEMANTIC')
  - next_token (string): Pagination cursor from a previous call
  - response_format ('markdown' | 'json'): Output format (default: 'markdown')

Returns for JSON format:
  {
    "passages": [
      {
        "content": string,        // Retrieved text chunk
        "score": number,          // Relevance score (0–1)
        "location": {             // Source location
          "type": string,         // "S3" | "WEB" | "CONFLUENCE" | etc.
          "uri": string           // Source URI
        },
        "metadata": object        // Optional key-value metadata
      }
    ],
    "next_token": string | null   // Cursor for next page (null if no more results)
  }

Examples:
  - "Find passages about return policy" → query="return policy", number_of_results=5
  - "Get top 10 results about shipping" → query="shipping", number_of_results=10, search_type="HYBRID"
  - "Next page of results" → query="shipping", next_token="<token from previous call>"

Error handling:
  - Returns "Error: Knowledge Base not found" if the KB ID is wrong
  - Returns "Error: Access denied" if the AWS profile lacks bedrock:Retrieve permission
  - Returns "Error: AWS request throttled" on rate limit`,
    inputSchema: RetrieveInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async (params: RetrieveInput) => {
    const kbId = params.knowledge_base_id ?? DEFAULT_KB_ID;
    if (!kbId) {
      return {
        content: [
          {
            type: "text",
            text: "Error: No Knowledge Base ID provided. Pass 'knowledge_base_id' or set the BEDROCK_KNOWLEDGE_BASE_ID environment variable.",
          },
        ],
        isError: true,
      };
    }

    try {
      const result = await retrieve({
        knowledgeBaseId: kbId,
        query: params.query,
        numberOfResults: params.number_of_results,
        searchType: params.search_type,
        nextToken: params.next_token,
      });

      if (result.passages.length === 0) {
        return {
          content: [{ type: "text", text: `No results found for query: "${params.query}"` }],
        };
      }

      const structured = {
        passages: result.passages.map((p) => ({
          content: p.content,
          score: p.score,
          location: p.location
            ? { type: p.location.type, uri: formatLocation(p.location) }
            : null,
          metadata: p.metadata ?? null,
        })),
        next_token: result.nextToken ?? null,
      };

      let text: string;
      if (params.response_format === "json") {
        text = JSON.stringify(structured, null, 2);
      } else {
        text = [
          `# Knowledge Base Results for: "${params.query}"`,
          `Found ${result.passages.length} passages${result.nextToken ? " (more available)" : ""}.`,
          "",
          passagesToMarkdown(result.passages),
          ...(result.nextToken
            ? ["", `---`, `*More results available. Pass \`next_token: "${result.nextToken}"\` to continue.*`]
            : []),
        ].join("\n");
      }

      const { text: finalText } = truncate(text);

      return {
        content: [{ type: "text", text: finalText }],
        structuredContent: structured,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: formatAwsError(error) }],
        isError: true,
      };
    }
  }
);

// ── Tool: bedrock_retrieve_and_generate ──────────────────────────────────────

const RetrieveAndGenerateInputSchema = z
  .object({
    query: z
      .string()
      .min(1, "Query must not be empty")
      .max(1000, "Query must not exceed 1000 characters")
      .describe("Natural language question to answer using the Knowledge Base"),
    knowledge_base_id: z
      .string()
      .optional()
      .describe(
        "Bedrock Knowledge Base ID. Defaults to BEDROCK_KNOWLEDGE_BASE_ID env var if not provided."
      ),
    model_arn: z
      .string()
      .default("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0")
      .describe(
        "ARN of the Bedrock foundation model used for generation (default: Claude 3 Haiku). Use a model available in your region."
      ),
    number_of_results: z
      .number()
      .int()
      .min(1)
      .max(100)
      .default(5)
      .describe("Number of passages to retrieve before generating (1–100, default: 5)"),
    search_type: z
      .enum(["SEMANTIC", "HYBRID"])
      .default("SEMANTIC")
      .describe("Search strategy: 'SEMANTIC' or 'HYBRID' (default: SEMANTIC)"),
    session_id: z
      .string()
      .optional()
      .describe(
        "Session ID from a previous call to continue a multi-turn conversation in the same KB session"
      ),
    system_prompt: z
      .string()
      .optional()
      .describe(
        "Optional custom system prompt template for the generation step. Use $search_results$ as placeholder for the retrieved content."
      ),
    response_format: z
      .enum(["markdown", "json"])
      .default("markdown")
      .describe("Output format: 'markdown' for human-readable, 'json' for structured data"),
  })
  .strict();

type RetrieveAndGenerateInput = z.infer<typeof RetrieveAndGenerateInputSchema>;

server.registerTool(
  "bedrock_retrieve_and_generate",
  {
    title: "Retrieve and Generate from Bedrock Knowledge Base",
    description: `Perform a full RAG (Retrieval-Augmented Generation) pipeline: retrieve relevant passages from a Bedrock Knowledge Base and then generate a grounded answer using a Bedrock foundation model.

Use this tool when you need a synthesized, natural language answer based on Knowledge Base content, with citations pointing back to the source passages.

Args:
  - query (string): Natural language question (required)
  - knowledge_base_id (string): Bedrock KB ID. Defaults to BEDROCK_KNOWLEDGE_BASE_ID env var.
  - model_arn (string): ARN of the Bedrock model to use for generation (default: Claude 3 Haiku)
  - number_of_results (number): Passages to retrieve before generating, 1–100 (default: 5)
  - search_type ('SEMANTIC' | 'HYBRID'): Search strategy (default: 'SEMANTIC')
  - session_id (string): Session ID from a previous call for multi-turn conversation
  - system_prompt (string): Custom generation prompt template (use $search_results$ placeholder)
  - response_format ('markdown' | 'json'): Output format (default: 'markdown')

Returns for JSON format:
  {
    "output": string,           // Generated answer grounded in KB content
    "session_id": string,       // Session ID for follow-up questions
    "citations": [
      {
        "generated_text": string,  // Span of generated text
        "sources": [               // Passages that support this span
          { "content": string, "location": { "type": string, "uri": string } }
        ]
      }
    ]
  }

Examples:
  - "What is the refund policy?" → query="refund policy"
  - "Continue conversation" → query="Can I also return accessories?", session_id="<id from previous>"
  - "Use Claude 3 Sonnet" → model_arn="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"

Error handling:
  - Returns "Error: Knowledge Base not found" if the KB ID is wrong
  - Returns "Error: Access denied" if missing bedrock:RetrieveAndGenerate permission
  - Returns "Error: Invalid request" if the model ARN is incorrect`,
    inputSchema: RetrieveAndGenerateInputSchema,
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  async (params: RetrieveAndGenerateInput) => {
    const kbId = params.knowledge_base_id ?? DEFAULT_KB_ID;
    if (!kbId) {
      return {
        content: [
          {
            type: "text",
            text: "Error: No Knowledge Base ID provided. Pass 'knowledge_base_id' or set the BEDROCK_KNOWLEDGE_BASE_ID environment variable.",
          },
        ],
        isError: true,
      };
    }

    try {
      const result = await retrieveAndGenerate({
        knowledgeBaseId: kbId,
        modelArn: params.model_arn,
        query: params.query,
        numberOfResults: params.number_of_results,
        searchType: params.search_type,
        sessionId: params.session_id,
        systemPrompt: params.system_prompt,
      });

      const structured = {
        output: result.output,
        session_id: result.sessionId ?? null,
        citations: result.citations.map((c) => ({
          generated_text: c.generatedText ?? null,
          sources: c.passages.map((p) => ({
            content: p.content,
            location: p.location
              ? { type: p.location.type, uri: formatLocation(p.location) }
              : null,
            metadata: p.metadata ?? null,
          })),
        })),
      };

      let text: string;
      if (params.response_format === "json") {
        text = JSON.stringify(structured, null, 2);
      } else {
        const citationBlock =
          result.citations.length > 0
            ? `\n\n## Citations\n\n${citationsToMarkdown(result.citations)}`
            : "";
        const sessionNote = result.sessionId
          ? `\n\n---\n*Session ID: \`${result.sessionId}\` — pass this as \`session_id\` to continue the conversation.*`
          : "";
        text = `# Answer\n\n${result.output}${citationBlock}${sessionNote}`;
      }

      const { text: finalText } = truncate(text);

      return {
        content: [{ type: "text", text: finalText }],
        structuredContent: structured,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: formatAwsError(error) }],
        isError: true,
      };
    }
  }
);

// ── Start ─────────────────────────────────────────────────────────────────────

async function main() {
  // Warn (but don't exit) if no default KB ID — users can still pass it per-call
  if (!DEFAULT_KB_ID) {
    console.error(
      "WARNING: BEDROCK_KNOWLEDGE_BASE_ID is not set. " +
        "You must pass 'knowledge_base_id' in every tool call, or set the env var."
    );
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Bedrock MCP Server running via stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
