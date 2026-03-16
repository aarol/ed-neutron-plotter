import { SearchBox } from "./SearchBox";

interface SearchBarProps {
  onSearch: (query: string) => void;
  onSuggest: (word: string) => string[];
  onClickRoute: (word: string) => void;
  onOpenJournal: () => void;
}

export function SearchBar({ onClickRoute, onOpenJournal, onSearch, onSuggest }: SearchBarProps) {
  return (
    <div className="pointer-events-auto fixed left-1/2 top-5 z-[1000] flex -translate-x-1/2 items-center gap-2">
      <SearchBox
        onClickRoute={onClickRoute}
        onSearch={onSearch}
        onSuggest={onSuggest}
        placeholder="Enter target star.."
      />

      <button
        aria-label="Track in-game location"
        className="flex h-10 w-10 shrink-0 items-center justify-center border border-white/30 bg-black/70 p-0 text-white/80 shadow-[0_4px_15px_rgba(0,0,0,0.2)] backdrop-blur-md transition hover:bg-white/15 hover:text-white"
        onClick={onOpenJournal}
        title="Track in-game location"
        type="button"
      >
        <svg className="h-5 w-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="8" />
          <line x1="12" x2="12" y1="2" y2="5" />
          <line x1="12" x2="12" y1="19" y2="22" />
          <line x1="2" x2="5" y1="12" y2="12" />
          <line x1="19" x2="22" y1="12" y2="12" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      </button>
    </div>
  );
}