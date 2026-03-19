export const uiTheme = {
  glassPanel:
    "border border-white/30 bg-black/70 text-space-text shadow-[0_12px_40px_rgba(0,0,0,0.6)] backdrop-blur-xl",
  panelHeader: "flex items-center justify-between",
  panelTitle: "m-0 text-sm font-semibold tracking-[0.02em]",
  iconButton:
    "flex h-8 w-8 items-center justify-center border border-white/25 bg-white/5 text-white/70 transition hover:bg-white/15 hover:text-white",
  ghostButton:
    "border border-white/30 bg-transparent px-3 py-1.5 text-xs font-medium text-white/75 transition hover:border-white/50 hover:bg-white/10",
  primaryButton:
    "border border-space-accent-strong bg-space-accent-strong px-3 py-1.5 text-xs font-semibold text-white shadow-[0_2px_10px_rgba(47,127,243,0.5)] transition hover:bg-space-accent hover:shadow-[0_4px_16px_rgba(47,127,243,0.62)] disabled:cursor-not-allowed disabled:opacity-60",
  textInput:
    "border border-white/30 bg-black/70 px-5 py-3 pr-11 text-base text-white shadow-[0_12px_40px_rgba(0,0,0,0.6)] outline-none backdrop-blur-xl transition placeholder:text-white/50 focus:border-white/60 focus:bg-black/80",
  compactInput:
    "w-full border border-white/30 bg-white/10 px-3 py-2 text-[13px] text-white outline-none transition placeholder:text-white/45 focus:border-white/55 focus:bg-white/15",
  suggestionsPanel:
    "absolute left-0 right-0 top-[calc(100%+8px)] z-[1001] max-h-[200px] overflow-y-auto border border-white/30 bg-black/90 backdrop-blur-md",
  suggestionRow:
    "cursor-pointer border-b border-white/10 px-5 py-2.5 text-white transition last:border-b-0 hover:bg-white/20",
};
