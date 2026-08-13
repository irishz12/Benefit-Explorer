export type Citation = {
  index: number;
  chunk_id: string;
  product: string;
  page: number;
  supporting_text: string;
};

export type CitationData = {
  citations: Citation[];
  detected_products: string[];
  retrieval_mode: string;
};

export type InsuranceMessagePart =
  | { type: "text"; text: string }
  | { type: "data-citations"; data: CitationData };

export type InsuranceMessage = {
  id: string;
  role: "user" | "assistant";
  parts: InsuranceMessagePart[];
};
