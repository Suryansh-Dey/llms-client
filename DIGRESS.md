# Where open-client departs from its sketches

## `Compactor`, the default `OverflowHandler` (0.4.0)

Sketched as an `ask()` holding a mutex, which summarizes in a spawned task once the history passes
half the context window. Option A of the turn-boundary question was agreed in chat.

| Sketched | Built | Why |
|---|---|---|
| `limit = 0.5 * context_window` | `Compactor::limit` in tokens, default 64,000 | The API doesn't report the window. At DeepSeek V4.1 Flash's 1M it would mean 524K tokens per request before the first summary. |
| `history.token_count()` | `usage.total_tokens` of the reply | Real count, as asked. A response without `usage` isn't compacted. |
| Summarize after the final answer, then pop until the last user prompt | Summarize the history *before* the last prompt, after any reply, tool calls included (option A) | A "summarize" message can't follow unanswered tool calls, and our bot rarely ends a turn without one (`offer_options`). The kept part is the same. |
| `tokio::spawn` + mutex unlocked when done | Awaited inside `ask()` / `stream()`, after the reply; `&mut Session` already serializes turns | Lambda freezes spawned tasks once the response is sent. A stream yields the whole reply first, then ends once the summary is in. |
| Summary goes into the system prompt | At the start of the kept user prompt, in `<summary_of_earlier_conversation>` tags | One `AI` serves every conversation, and handlers only see history. The system prompt and tools also stay cached across summaries. |
| `history.append("summarize")`, `ask`, `pop_back` | A separate request: the history before the last prompt, then the prompt, with `tool_choice: none` and no `response_format` | Its start is the request just sent, so it's served from cache. With tools turned off the model can't call `retrieve` instead, and it answers as text. |
| (Replace `SlidingWindow` as the default) | `SlidingWindow` removed | As asked. Tests and the token budget example use small local handlers. |

### Added beyond the ask
- `AI::include_usage()` defaults to true, since streams otherwise carry no `usage` and the default handler would never run on them.
- `AI::stream()` borrows the `AI` for the stream's lifetime (`&'a self`), so the handler can ask it for the summary.
- `Compactor::prompt` can be replaced. It isn't serialized while it's the default, so stored sessions stay small.
- The handler gets the client privately, through `HistoryTrimmer`. There is no public accessor.

### Left out
- A failed or empty summary isn't reported: nothing is forgotten, and it's retried after the next reply.
- Replies added with `Session::add_completion()` aren't compacted, as there's no client to ask (WASM without `reqwest`).

### Breaking (0.3 → 0.4)
- `SlidingWindow` is gone.
- `Session` and `ChatStream` default to `Compactor`.
- `AI::stream(&'a self, ..)`.
- `stream_options.include_usage` is in the default params.

### Verified
- 4 mock-server tests cover:
  - what is kept and the rolling summary;
  - that the summary request starts with the previous request, with tools off and no `response_format`;
  - tool calls of the last turn;
  - failures, empty summaries and no client;
  - the summary finishing before a stream ends.
- A live 3-turn stream on `deepseek/deepseek-v4.1-flash-1017` via OpenRouter with `limit` 150:
  - usage arrived by default;
  - `tool_choice: none` with a tool was accepted;
  - the second summary folded in the first;
  - the third turn answered "name, program, semester" from the summary alone.
  
  Caching wasn't observable: the prompts (about 400 tokens) were below the providers' minimum. The sentence "leaving out what didn't come up" was added to the prompt after that run.
