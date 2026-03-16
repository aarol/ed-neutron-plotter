import { SearchBox } from "./SearchBox";
import styles from "./UI.module.css";

interface SearchBarProps {
  onSearch: (query: string) => void;
  onSuggest: (word: string) => string[];
  onClickRoute: (word: string) => void;
  onOpenJournal: () => void;
}

export function SearchBar({ onClickRoute, onOpenJournal, onSearch, onSuggest }: SearchBarProps) {
  return (
    <div className={styles.searchWrapper}>
      <SearchBox
        onClickRoute={onClickRoute}
        onSearch={onSearch}
        onSuggest={onSuggest}
        placeholder="Enter target star.."
      />

      <button
        aria-label="Track in-game location"
        className={styles.gpsIcon}
        onClick={onOpenJournal}
        title="Track in-game location"
        type="button"
      >
        <svg fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
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