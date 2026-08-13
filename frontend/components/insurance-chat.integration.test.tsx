import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { InsuranceChat } from "@/components/insurance-chat";

const resultEvent = {
  event: "result",
  answer: "Kotak TULIP pays a maturity benefit [1].",
  citations: [
    {
      index: 1,
      chunk_id: "chunk_tulip",
      product: "Kotak TULIP",
      page: 4,
      supporting_text: "The maturity benefit shall be paid.",
    },
  ],
  detected_products: ["Kotak TULIP"],
  retrieval_mode: "hard_filter",
};

function streamResponse(events: object[]) {
  const payload = `${events.map((event) => JSON.stringify(event)).join("\n")}\n`;
  return new Response(payload, {
    status: 200,
    headers: { "Content-Type": "application/x-ndjson" },
  });
}

describe("InsuranceChat streaming flow", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders a streamed answer and its verified source", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      streamResponse([
        { event: "status", message: "Searching product brochures…" },
        { event: "answer_delta", delta: "Kotak TULIP pays a maturity benefit [1]." },
        resultEvent,
      ]),
    );
    render(<InsuranceChat />);

    fireEvent.click(
      screen.getByRole("button", {
        name: "What maturity benefit does Kotak TULIP pay?",
      }),
    );

    expect(
      await screen.findByText(/Kotak TULIP pays a maturity benefit/),
    ).toBeInTheDocument();
    expect(screen.getByText("Verified sources")).toBeInTheDocument();
    expect(screen.getByText("Kotak TULIP")).toBeInTheDocument();
    expect(screen.getByText("Page 4")).toBeInTheDocument();
  });

  it("shows a readable error when the API is unavailable", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: "Backend unavailable" }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      }),
    );
    render(<InsuranceChat />);

    fireEvent.change(
      screen.getByPlaceholderText(
        "Ask about a product, benefit, exclusion, or comparison…",
      ),
      { target: { value: "What is covered?" } },
    );
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));

    await waitFor(() => {
      expect(screen.getByText("Backend unavailable")).toBeInTheDocument();
    });
  });
});
