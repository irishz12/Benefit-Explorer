export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = await request.json();
  const backendUrl = process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000";
  try {
    const response = await fetch(`${backendUrl}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: body.message }),
      cache: "no-store",
      signal: request.signal,
    });
    if (!response.ok || !response.body) {
      const detail = await response.text();
      return Response.json(
        { error: detail || "The insurance assistant is unavailable." },
        { status: response.status || 502 },
      );
    }
    return new Response(response.body, {
      status: 200,
      headers: {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return Response.json(
      { error: "Cannot reach the Python backend. Start it on port 8000 and try again." },
      { status: 502 },
    );
  }
}
