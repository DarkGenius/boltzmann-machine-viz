interface TermHighlightProps {
  term: string;
  definition: string;
}

export function TermHighlight({ term, definition }: TermHighlightProps) {
  return (
    <span className="term-highlight" data-tooltip={definition}>
      {term}
    </span>
  );
}