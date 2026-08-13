import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CitationText } from "@/components/citation-text";
import { SourcePanel } from "@/components/source-panel";
import type { CitationData } from "@/lib/types";

const data: CitationData = {
  citations: [
    {
      index: 1,
      chunk_id: "chunk_1",
      product: "Kotak EDGE",
      page: 2,
      supporting_text: "The policy pays the stated benefit.",
    },
  ],
  detected_products: ["Kotak EDGE"],
  retrieval_mode: "hard_filter",
};

describe("citation source links", () => {
  it("namespaces source identifiers by message", () => {
    const { container } = render(
      <>
        <SourcePanel data={data} sourceIdPrefix="message-a" />
        <SourcePanel data={data} sourceIdPrefix="message-b" />
      </>,
    );
    expect(container.querySelector("#message-a-source-1")).not.toBeNull();
    expect(container.querySelector("#message-b-source-1")).not.toBeNull();
  });

  it("scrolls to the source belonging to the same message", () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;

    render(
      <>
        <CitationText
          text="Grounded claim [1]"
          citations={data.citations}
          sourceIdPrefix="message-b"
        />
        <SourcePanel data={data} sourceIdPrefix="message-b" />
      </>,
    );
    fireEvent.click(screen.getByRole("button", { name: "[1]" }));
    expect(scrollIntoView).toHaveBeenCalledOnce();
  });
});
