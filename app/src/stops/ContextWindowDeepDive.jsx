// Bonus stop: Deep dive into context windows, context rot, and management strategies.
// Accessible from Explore mode via the "Context window is finite" aside in RAGSimple.

export default function ContextWindowDeepDive() {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <span className="text-xs font-semibold text-purple-400 uppercase tracking-wide">
          Bonus Deep Dive
        </span>
        <h2 className="text-xl font-bold text-white mt-1">
          The Context Window: What It Is, Why It Breaks, and How to Manage It
        </h2>
        <p className="text-sm text-slate-400 mt-2">
          Every interaction with a language model happens inside a fixed-size window of text.
          Understanding how that window works &mdash; and fails &mdash; is essential to building
          reliable AI systems.
        </p>
      </div>

      {/* What is a context window */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-purple-400 mb-3">What is a context window?</h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          A context window is the total amount of text a model can "see" at once &mdash; your system
          prompt, the conversation history, any injected documents (like RAG results), and the
          model's own response, all concatenated into a single sequence of tokens.
        </p>
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/30 font-mono text-xs text-slate-400 space-y-1">
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-purple-400">System</span>
            <div className="flex-1 h-5 bg-purple-900/40 rounded flex items-center px-2 text-purple-300">
              "You are a storage I/O expert..."
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-yellow-400">RAG docs</span>
            <div className="flex-1 h-5 bg-yellow-900/30 rounded flex items-center px-2 text-yellow-300">
              Retrieved reference patterns
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-blue-400">History</span>
            <div className="flex-1 h-5 bg-blue-900/30 rounded flex items-center px-2 text-blue-300">
              Prior messages in this conversation
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-green-400">Query</span>
            <div className="flex-1 h-5 bg-green-900/30 rounded flex items-center px-2 text-green-300">
              "Classify this I/O pattern: IOPS: 45000..."
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-slate-500">Response</span>
            <div className="flex-1 h-5 bg-slate-800 rounded border border-dashed border-slate-600 flex items-center px-2 text-slate-500">
              Model generates into remaining space
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-700/50 text-slate-500 text-center">
            Total: all of the above must fit within the model's token limit
          </div>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-slate-400">
          <div className="p-2 rounded bg-slate-900/30">
            <span className="font-semibold text-slate-300">SmolLM2-360M</span>
            <br />
            2,048 tokens
          </div>
          <div className="p-2 rounded bg-slate-900/30">
            <span className="font-semibold text-slate-300">GPT-4o / Claude</span>
            <br />
            128K&ndash;200K tokens
          </div>
          <div className="p-2 rounded bg-slate-900/30">
            <span className="font-semibold text-slate-300">Gemini 1.5 Pro</span>
            <br />
            1M+ tokens
          </div>
        </div>
      </div>

      {/* How context is built */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-purple-400 mb-3">
          How context is built and managed
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          The context window isn't just a text buffer &mdash; it's actively managed by the
          application layer. Each component competes for space:
        </p>
        <div className="space-y-3">
          <div className="flex gap-3 items-start">
            <span className="text-xs font-mono text-purple-400 w-28 shrink-0 pt-0.5">
              System prompt
            </span>
            <p className="text-xs text-slate-400">
              Fixed instructions that define the model's role and behavior. Usually 200&ndash;1,000
              tokens. Set once per conversation and always occupies the same space.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-xs font-mono text-yellow-400 w-28 shrink-0 pt-0.5">
              RAG injection
            </span>
            <p className="text-xs text-slate-400">
              Retrieved documents are inserted before the user's query. The more documents you
              retrieve (higher top-K), the more tokens consumed. A typical RAG setup might inject
              500&ndash;2,000 tokens of context per query.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-xs font-mono text-blue-400 w-28 shrink-0 pt-0.5">
              Conversation
            </span>
            <p className="text-xs text-slate-400">
              Every prior message &mdash; user and assistant &mdash; stays in the window. This is
              what gives the model "memory" within a session. But it grows with every exchange,
              eventually crowding out everything else.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-xs font-mono text-orange-400 w-28 shrink-0 pt-0.5">Few-shot</span>
            <p className="text-xs text-slate-400">
              Examples injected to demonstrate the desired format. Each example might be
              100&ndash;300 tokens. Three examples = 300&ndash;900 tokens consumed before the model
              even sees the real query.
            </p>
          </div>
        </div>
      </div>

      {/* Context rot */}
      <div className="p-5 rounded-lg bg-red-950/20 border border-red-800/30">
        <h3 className="text-base font-semibold text-red-400 mb-3">
          Context rot: when more context makes things worse
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          Intuitively, more context should help. In practice, it often hurts. As the context window
          fills up, models exhibit several failure modes:
        </p>
        <div className="space-y-3">
          <div className="p-3 rounded bg-slate-800/50">
            <h4 className="text-sm font-semibold text-red-300 mb-1">Lost in the middle</h4>
            <p className="text-xs text-slate-400">
              Research shows that models pay the most attention to the <em>beginning</em> and
              <em>end</em> of the context window. Information placed in the middle is
              disproportionately ignored. If your most relevant RAG document lands in the middle of
              10 retrieved passages, the model may miss it entirely.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50">
            <h4 className="text-sm font-semibold text-red-300 mb-1">Attention dilution</h4>
            <p className="text-xs text-slate-400">
              The self-attention mechanism distributes "attention" across all tokens in the window.
              As the window grows, each token gets a thinner slice of attention. Important details
              that were prominent in a short context become noise in a long one. This is why a
              concise, well-structured prompt often outperforms a verbose one with more information.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50">
            <h4 className="text-sm font-semibold text-red-300 mb-1">Contradictory context</h4>
            <p className="text-xs text-slate-400">
              Long conversations accumulate stale information. If the user corrected themselves
              three messages ago, both the wrong and right answers are in the window. The model must
              figure out which is current &mdash; and it doesn't always get it right. RAG can make
              this worse by retrieving documents that contradict each other.
            </p>
          </div>
        </div>
      </div>

      {/* Management strategies */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-purple-400 mb-3">
          Strategies for context management
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wide mb-2">
              Sliding window
            </h4>
            <p className="text-xs text-slate-400">
              Drop the oldest messages when the window fills up. Simple but lossy &mdash; the model
              forgets early instructions and context. Works for casual chat, dangerous for
              multi-step tasks.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wide mb-2">
              Summarization / compaction
            </h4>
            <p className="text-xs text-slate-400">
              Use the model itself (or a smaller model) to summarize older messages before
              discarding them. Preserves key facts in fewer tokens. But summaries are lossy too
              &mdash; the model decides what's "important," and it may be wrong.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wide mb-2">
              Structured memory
            </h4>
            <p className="text-xs text-slate-400">
              Maintain a separate structured store (key-value pairs, task state, user preferences)
              that gets injected into the system prompt each turn. Only the current state is in the
              window &mdash; not the full history of how it got there.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wide mb-2">
              RAG as context relief
            </h4>
            <p className="text-xs text-slate-400">
              Instead of keeping everything in the window, store knowledge externally and retrieve
              only what's needed per query. This is one of RAG's underappreciated benefits: it lets
              you keep the window lean while still having access to a large knowledge base.
            </p>
          </div>
        </div>
      </div>

      {/* Connection to post-training */}
      <div className="p-4 rounded-lg bg-gradient-to-r from-purple-950/30 to-slate-800 border border-purple-800/30">
        <h4 className="text-sm font-semibold text-purple-400 mb-2">The post-training connection</h4>
        <p className="text-sm text-slate-300 leading-relaxed mb-2">
          Context management is an inference-time problem &mdash; you're engineering around the
          model's limitations. Post-training attacks the problem from the other direction: if the
          model <em>already knows</em> your domain, you need less context to get a good answer.
        </p>
        <p className="text-sm text-slate-400 leading-relaxed">
          A fine-tuned model doesn't need few-shot examples (it learned the format during training).
          It may need fewer RAG documents (it has internalized the domain knowledge). The context
          window is still finite, but a well-trained model uses it more efficiently.
        </p>
      </div>
    </div>
  )
}
