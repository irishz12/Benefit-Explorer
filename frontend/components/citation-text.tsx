"use client";

import type { Citation } from "@/lib/types";

export function CitationText({
  text,
  citations,
  sourceIdPrefix,
}: {
  text: string;
  citations: Citation[];
  sourceIdPrefix: string;
}) {
  const citationMap = new Map(citations.map((citation) => [citation.index, citation]));
  const pieces = text.split(/(\[\d+\])/g);
  return (
    <div className="whitespace-pre-wrap text-[15px] leading-7 text-stone-700">
      {pieces.map((piece, index) => {
        const match = piece.match(/^\[(\d+)\]$/);
        if (!match) return <span key={index}>{piece}</span>;
        const citation = citationMap.get(Number(match[1]));
        return (
          <button
            key={`${piece}-${index}`}
            type="button"
            disabled={!citation}
            onClick={() =>
              document
                .getElementById(`${sourceIdPrefix}-source-${citation?.index}`)
                ?.scrollIntoView({ behavior: "smooth", block: "center" })
            }
            title={citation ? `${citation.product}, page ${citation.page}` : "Source"}
            className="mx-0.5 inline-flex translate-y-[-1px] items-center rounded-md bg-emerald-100 px-1.5 py-0.5 text-xs font-bold text-emerald-800 hover:bg-emerald-200 disabled:opacity-60"
          >
            {piece}
          </button>
        );
      })}
    </div>
  );
}
