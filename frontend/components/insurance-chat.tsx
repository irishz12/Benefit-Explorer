"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  Bot,
  CircleAlert,
  MessageSquareText,
  RotateCcw,
  Send,
  User,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { CitationText } from "@/components/citation-text";
import { SourcePanel } from "@/components/source-panel";
import type { CitationData, InsuranceMessage } from "@/lib/types";

const SUGGESTIONS = [
  "What maturity benefit does Kotak TULIP pay?",
  "Compare EDGE and TULIP surrender benefits",
  "How is Sum Assured on Death determined under Kotak GAIN?",
];

function messageText(message: InsuranceMessage) {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");
}

function citationData(message: InsuranceMessage): CitationData | undefined {
  const part = message.parts.find((item) => item.type === "data-citations");
  return part?.data;
}

function newMessage(role: "user" | "assistant", text = ""): InsuranceMessage {
  return { id: crypto.randomUUID(), role, parts: [{ type: "text", text }] };
}

export function InsuranceChat() {
  const [messages, setMessages] = useState<InsuranceMessage[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  async function ask(rawQuestion: string) {
    const question = rawQuestion.trim();
    if (!question || isLoading) return;
    const userMessage = newMessage("user", question);
    const assistantMessage = newMessage("assistant");
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    setError(null);
    setStatus("Connecting to the knowledge base…");
    setIsLoading(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(
          payload.error ?? "The assistant could not answer this question.",
        );
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const answerDeltas: string[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line) as Record<string, unknown>;
          if (event.event === "status") setStatus(String(event.message));
          if (event.event === "answer_delta") {
            answerDeltas.push(String(event.delta));
            setStatus("");
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessage.id
                  ? {
                      ...message,
                      parts: [{ type: "text", text: answerDeltas.join("") }],
                    }
                  : message,
              ),
            );
          }
          if (event.event === "result") {
            const data: CitationData = {
              citations: event.citations as CitationData["citations"],
              detected_products: event.detected_products as string[],
              retrieval_mode: String(event.retrieval_mode),
            };
            const finalAnswer = String(event.answer);
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessage.id
                  ? {
                      ...message,
                      parts: [
                        { type: "text", text: finalAnswer },
                        { type: "data-citations", data },
                      ],
                    }
                  : message,
              ),
            );
          }
          if (event.event === "error") throw new Error(String(event.message));
        }
      }
    } catch (caught) {
      if ((caught as Error).name !== "AbortError") {
        const message =
          caught instanceof Error ? caught.message : "Something went wrong.";
        setError(message);
        setMessages((current) =>
          current.filter((item) => item.id !== assistantMessage.id),
        );
      }
    } finally {
      setIsLoading(false);
      setStatus("");
      abortRef.current = null;
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void ask(input);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void ask(input);
    }
  }

  return (
    <main className="flex min-h-dvh flex-col bg-[#f7f7f4]">
      <header className="sticky top-0 z-20 border-b border-stone-200/80 bg-[#f7f7f4]/90 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 md:px-6">
          <div>
            <div className="font-semibold tracking-tight text-stone-900">
              BenefitExplorer
            </div>
            <div className="hidden text-xs text-stone-500 sm:block">
              Insurance Answers Grounded in Brochures
            </div>
          </div>
          <div className="flex items-center gap-1">
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  abortRef.current?.abort();
                  setMessages([]);
                  setError(null);
                }}
              >
                <RotateCcw className="size-3.5" />
                New chat
              </Button>
            )}
          </div>
        </div>
      </header>

      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-4 md:px-6">
        {messages.length === 0 ? (
          <section className="flex flex-1 flex-col items-center justify-center py-16 text-center">
            <p className="mb-3 text-xs font-bold uppercase tracking-[0.2em] text-emerald-700">
              Verified brochure sources
            </p>
            <h1 className="max-w-2xl text-3xl font-semibold tracking-[-0.04em] text-stone-900 sm:text-5xl">
              Understand your policy, without the fine-print fatigue.
            </h1>
            <p className="mt-5 max-w-xl text-sm leading-6 text-stone-600 sm:text-base">
              Ask about benefits, exclusions, eligibility, waiting periods,
              surrender terms—or compare two products. Every answer includes
              verified page-level sources.
            </p>
            <div className="mt-8 grid w-full max-w-3xl gap-3 md:grid-cols-3">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => void ask(suggestion)}
                  className="rounded-2xl border border-stone-200 bg-white p-4 text-left text-sm leading-5 text-stone-700 shadow-sm transition hover:-translate-y-0.5 hover:border-emerald-300 hover:shadow-md"
                >
                  <MessageSquareText className="mb-3 size-4 text-emerald-700" />
                  {suggestion}
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section className="mx-auto w-full max-w-4xl flex-1 py-8 md:py-12">
            <div className="space-y-8">
              {messages.map((message) => {
                const text = messageText(message);
                const sources = citationData(message);
                const assistant = message.role === "assistant";
                return (
                  <article
                    key={message.id}
                    className={
                      assistant
                        ? "grid gap-4 md:grid-cols-[42px_1fr]"
                        : "flex justify-end"
                    }
                  >
                    {assistant ? (
                      <>
                        <div className="flex size-9 items-center justify-center rounded-xl bg-emerald-800 text-white">
                          <Bot className="size-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
                            BenefitExplorer
                          </div>
                          {text ? (
                            <CitationText
                              text={text}
                              citations={sources?.citations ?? []}
                              sourceIdPrefix={message.id}
                            />
                          ) : (
                            <div className="flex items-center gap-2 text-sm text-stone-500">
                              <span className="flex gap-1">
                                <i className="size-1.5 animate-bounce rounded-full bg-emerald-600" />
                                <i className="size-1.5 animate-bounce rounded-full bg-emerald-600 [animation-delay:120ms]" />
                                <i className="size-1.5 animate-bounce rounded-full bg-emerald-600 [animation-delay:240ms]" />
                              </span>
                              {status}
                            </div>
                          )}
                          {isLoading &&
                            message.id === messages.at(-1)?.id &&
                            text && <span className="streaming-cursor" />}
                          {sources && (
                            <div className="mt-5">
                              <SourcePanel
                                data={sources}
                                sourceIdPrefix={message.id}
                              />
                            </div>
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="flex max-w-[88%] items-start gap-3 rounded-2xl rounded-tr-sm bg-stone-900 px-4 py-3 text-sm leading-6 text-white sm:max-w-[75%]">
                        <User className="mt-0.5 size-4 shrink-0 text-stone-400" />
                        {text}
                      </div>
                    )}
                  </article>
                );
              })}
              {error && (
                <div className="flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                  <CircleAlert className="mt-0.5 size-4 shrink-0" />
                  <div>
                    <div className="font-semibold">
                      Couldn’t complete that answer
                    </div>
                    <div className="mt-1 text-red-700">{error}</div>
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>
          </section>
        )}
      </div>

      <footer className="sticky bottom-0 z-10 border-t border-stone-200/80 bg-[#f7f7f4]/95 px-4 py-3 backdrop-blur-xl md:px-6 md:py-5">
        <form onSubmit={submit} className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2 rounded-2xl border border-stone-300 bg-white p-2 shadow-lg shadow-stone-200/60 focus-within:border-emerald-500 focus-within:ring-2 focus-within:ring-emerald-100">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={1}
              placeholder="Ask about a product, benefit, exclusion, or comparison…"
              className="max-h-36 min-h-11 flex-1 resize-none bg-transparent px-3 py-2.5 text-sm leading-6 text-stone-900 outline-none placeholder:text-stone-400 disabled:opacity-60"
            />
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || isLoading}
              aria-label="Send question"
            >
              <Send className="size-4" />
            </Button>
          </div>
          <p className="mt-2 text-center text-[11px] text-stone-400">
            Answers are based on indexed brochures. Verify policy terms before
            making financial decisions.
          </p>
        </form>
      </footer>
    </main>
  );
}
