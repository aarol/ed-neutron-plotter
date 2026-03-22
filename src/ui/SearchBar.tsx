import { useContext } from "preact/hooks";
import { SearchBox } from "./SearchBox";
import type { SearchSubmitOptions } from "./SearchBox";
import { Button } from "./components/Button";
import { JournalContext } from "./state/journalModel";

interface SearchBarProps {
  onSearch: (query: string, options: SearchSubmitOptions) => void;
  onSuggest: (word: string) => string[];
  onClickRoute: (word: string) => void;
  onOpenJournal: () => void;
}

export function SearchBar({ onClickRoute, onOpenJournal, onSearch, onSuggest }: SearchBarProps) {
  const journalState = useContext(JournalContext)!;
  const isJournalTracking = journalState.enabled;

  return (
    <div className="pointer-events-auto fixed left-1/2 top-5 z-1000 flex -translate-x-1/2 items-center">
      <SearchBox
        onClickRoute={onClickRoute}
        onSearch={onSearch}
        onSuggest={onSuggest}
        placeholder="Enter target star.."
        showRouteButton={false}
        rightIcon={
          <Button
            aria-label="Track in-game location"
            className={`flex h-full w-full items-center justify-center border border-transparent p-0 ${
              isJournalTracking.value
                ? "bg-space-accent-strong/20 text-space-accent drop-shadow-[0_0_10px_rgba(106,166,255,0.75)] hover:bg-space-accent-strong/30"
                : "bg-transparent text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.6)] hover:bg-white/10"
            }`.trim()}
            onClick={onOpenJournal}
            title={isJournalTracking.value ? "Stop tracking in-game location" : "Track in-game location"}
            variant="plain"
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.2" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="8" />
              <line x1="12" x2="12" y1="2" y2="5" />
              <line x1="12" x2="12" y1="19" y2="22" />
              <line x1="2" x2="5" y1="12" y2="12" />
              <line x1="19" x2="22" y1="12" y2="12" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </Button>
        }
      />
    </div>
  );
}