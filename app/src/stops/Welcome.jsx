export default function Welcome() {
  return (
    <div className="max-w-3xl mx-auto text-center">
      <div className="grid grid-cols-2 gap-4 text-left max-w-xl mx-auto">
        <div className="p-3 rounded-lg bg-yellow-950/20 border border-yellow-800/30 min-h-[80px]">
          <h4 className="text-sm font-semibold text-yellow-400 mb-1 whitespace-nowrap">RAG</h4>
          <p className="text-xs text-slate-400">Give the model knowledge at inference time via retrieval.</p>
        </div>
        <div className="p-3 rounded-lg bg-cyan-950/20 border border-cyan-800/30 min-h-[80px]">
          <h4 className="text-sm font-semibold text-cyan-400 mb-1 whitespace-nowrap">All the Options</h4>
          <p className="text-xs text-slate-400">Combine context optimization with model optimization.</p>
        </div>
        <div className="p-3 rounded-lg bg-orange-950/20 border border-orange-800/30 min-h-[80px]">
          <h4 className="text-sm font-semibold text-orange-400 mb-1 whitespace-nowrap">Prompt Engineering</h4>
          <p className="text-xs text-slate-400">Shape behavior through the prompt. No changes to the model.</p>
        </div>
        <div className="p-3 rounded-lg bg-slate-800 border border-slate-600/30 min-h-[80px]">
          <h4 className="text-sm font-semibold text-slate-200 mb-1 whitespace-nowrap">Post Training</h4>
          <p className="text-xs text-slate-400">Change the model itself: SFT, DPO, GRPO. This is the focus.</p>
        </div>
      </div>
    </div>
  )
}
