# aws-bedrock-mcp-server

MCP server for AWS Bedrock Knowledge Base — exposes two tools for semantic search and RAG.

| Tool | Description |
|------|-------------|
| `bedrock_retrieve` | Semantic / hybrid search — returns raw passages from the Knowledge Base |
| `bedrock_retrieve_and_generate` | Full RAG pipeline — retrieves passages **and** generates a grounded answer |

---

## Usage with Claude Code

```bash
claude mcp add aws-bedrock -- npx -y @miguel-carrera/aws-bedrock-mcp-server
```

Or add to your `claude_desktop_config.json` / `.mcp.json`:

```json
{
  "mcpServers": {
    "aws-bedrock": {
      "command": "npx",
      "args": ["-y", "@miguel-carrera/aws-bedrock-mcp-server"],
      "env": {
        "BEDROCK_KNOWLEDGE_BASE_ID": "<your-kb-id>",
        "AWS_REGION": "us-east-1",
        "AWS_PROFILE": "default"
      }
    }
  }
}
```

---

## Prerequisites

- Node.js ≥ 18
- AWS credentials configured in `~/.aws/credentials` or `~/.aws/config`

Required IAM permissions:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:RetrieveAndGenerate"
  ],
  "Resource": "arn:aws:bedrock:*:*:knowledge-base/*"
}
```

---

## Configuration

| Environment variable | Required | Default | Description |
|---|---|---|---|
| `BEDROCK_KNOWLEDGE_BASE_ID` | Recommended | — | Default Knowledge Base ID; can be overridden per tool call |
| `AWS_REGION` | No | `us-east-1` | AWS region where your Knowledge Base lives |
| `AWS_PROFILE` | No | `default` | AWS named profile from `~/.aws` |

---

## Tools

### `bedrock_retrieve`

Search the Knowledge Base and return raw passages.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | — | Natural language search query (required) |
| `knowledge_base_id` | string | env var | Bedrock KB ID |
| `number_of_results` | number | 5 | Passages to return (1–100) |
| `search_type` | `SEMANTIC` \| `HYBRID` | `SEMANTIC` | Search strategy |
| `next_token` | string | — | Pagination cursor |
| `response_format` | `markdown` \| `json` | `markdown` | Output format |

### `bedrock_retrieve_and_generate`

Retrieve passages and generate a grounded answer using a Bedrock model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | — | Question to answer (required) |
| `knowledge_base_id` | string | env var | Bedrock KB ID |
| `model_arn` | string | Claude 3 Haiku | ARN of the Bedrock model for generation |
| `number_of_results` | number | 5 | Passages to retrieve (1–100) |
| `search_type` | `SEMANTIC` \| `HYBRID` | `SEMANTIC` | Search strategy |
| `session_id` | string | — | Session ID for multi-turn conversations |
| `system_prompt` | string | — | Custom generation prompt (use `$search_results$` placeholder) |
| `response_format` | `markdown` \| `json` | `markdown` | Output format |

---

## Development

```bash
npm install
npm run dev      # watch mode
npm run build    # compile to dist/
```
