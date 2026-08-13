"use client";

import { BookOpenText, ExternalLink, FileText } from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import type { CitationData } from "@/lib/types";

export function SourcePanel({
  data,
  sourceIdPrefix,
}: {
  data: CitationData;
  sourceIdPrefix: string;
}) {
  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-4 shadow-sm md:p-5">
      <div className="mb-2 flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-stone-900">
            <BookOpenText className="size-4 text-emerald-700" /> Verified sources
          </div>
          <p className="mt-1 text-xs text-stone-500">
            {data.detected_products.length
              ? `Matched ${data.detected_products.join(" & ")}`
              : "Matched across the brochure library"}
          </p>
        </div>
        <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-emerald-700">
          {data.citations.length} {data.citations.length === 1 ? "source" : "sources"}
        </span>
      </div>
      <Accordion type="multiple" defaultValue={data.citations.map((citation) => String(citation.index))}>
        {data.citations.map((citation) => (
          <AccordionItem
            key={`${citation.index}-${citation.chunk_id}`}
            value={String(citation.index)}
            id={`${sourceIdPrefix}-source-${citation.index}`}
          >
            <AccordionTrigger>
              <span className="flex min-w-0 items-center gap-3">
                <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-stone-100 text-xs font-bold text-stone-700">{citation.index}</span>
                <span className="min-w-0">
                  <span className="block truncate text-sm font-semibold text-stone-900">{citation.product}</span>
                  <span className="flex items-center gap-1 text-xs text-stone-500"><FileText className="size-3" /> Page {citation.page}</span>
                </span>
              </span>
            </AccordionTrigger>
            <AccordionContent>
              <blockquote className="border-l-2 border-emerald-300 pl-3 text-sm leading-6 text-stone-600">
                “{citation.supporting_text}”
              </blockquote>
              <div className="mt-3 flex items-center gap-1 text-[11px] text-stone-400">
                <ExternalLink className="size-3" /> {citation.chunk_id}
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </section>
  );
}
