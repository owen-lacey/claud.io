"use client";

import React, { useEffect, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import SquadPanel from "@/components/team/SquadPanel";
import TransferSuggester from "@/components/team/TransferSuggester";
import ExplainPicks from "@/components/team/ExplainPicks";
import SavedSquadsPanel from "@/components/team/SavedSquadsPanel";
import { Button } from "@/components/ui/button";
import { ToastContainer, toastManager } from "@/components/ui/toast";
import { persistenceService } from "@/lib/persistence";

export default function AssistantPage() {
  const api = "/api/chat-tools";
  const [toolSquad, setToolSquad] = useState<any | null>(null);
  const [toolTransfers, setToolTransfers] = useState<any[] | null>(null);
  const [loadingBuild, setLoadingBuild] = useState(false);
  const [loadingSuggest, setLoadingSuggest] = useState(false);

  // Initialize state from persistence on mount
  useEffect(() => {
    const session = persistenceService.getSessionData();
    setToolSquad(session.conversationState.toolSquad);
    setToolTransfers(session.conversationState.toolTransfers);
  }, []);

  // Auto-save conversation state when tool results change
  useEffect(() => {
    persistenceService.updateConversationState({
      toolSquad,
      toolTransfers,
    });
  }, [toolSquad, toolTransfers]);

  // Attach auth headers for chat-tools calls and tap the stream to capture tool results
  useEffect(() => {
    const originalFetch = window.fetch;
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : "";
      const isChatTools = url.startsWith(api);

      if (isChatTools) {
        const token = (() => {
          try {
            return localStorage.getItem("pl_profile")?.replace(/"/g, "");
          } catch {
            return undefined;
          }
        })();
        const nextInit: RequestInit = { ...(init || {}) };
        const headers = new Headers(nextInit.headers || {});
        if (token) {
          headers.set("pl_profile", token);
          headers.set("Authorization", `Bearer ${token}`);
        }
        nextInit.headers = headers;

        const res = await originalFetch(input, nextInit);
        try {
          if (!res.body) return res;
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          const stream = new ReadableStream<Uint8Array>({
            async start(controller) {
              while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                if (value) {
                  const text = decoder.decode(value);
                  buffer += text;
                  let idx = buffer.indexOf("\n");
                  while (idx >= 0) {
                    const line = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 1);
                    // Frame parsing: only parse action/data frames
                    if (line.startsWith("a:") || line.startsWith("d:")) {
                      const payload = line.slice(2).trim();
                      try {
                        const obj = JSON.parse(payload);
                        const maybe = obj?.selectedSquad || obj?.squad;
                        if (maybe) setToolSquad(maybe);
                      } catch {
                        // ignore invalid JSON payloads
                      }
                    } else if (line.startsWith("3:")) {
                      // Error frame; optionally log for debugging
                      // no-op for now
                    }
                    idx = buffer.indexOf("\n");
                  }
                  controller.enqueue(value);
                }
              }
              controller.close();
            },
          });

          return new Response(stream, {
            status: res.status,
            headers: res.headers,
          });
        } catch {
          return res;
        }
      }

      return originalFetch(input, init);
    };
    return () => {
      window.fetch = originalFetch;
    };
  }, [api]);

  const runtime = useChatRuntime({ api });

  const getAuthHeaders = () => {
    let token: string | undefined = undefined;
    try { token = localStorage.getItem("pl_profile")?.replace(/"/g, ""); } catch {}
    const headers = new Headers();
    headers.set("content-type", "application/json");
    if (token) {
      headers.set("pl_profile", token);
      headers.set("Authorization", `Bearer ${token}`);
    }
    return headers;
  };

  const wrapSquadIfNeeded = (rawSquad: any) => {
    const looksBackend = Array.isArray(rawSquad?.startingXi) && rawSquad.startingXi[0]?.player;
    return looksBackend ? rawSquad : { selectedSquad: rawSquad };
  };

  const onBuildSquad = async () => {
    setLoadingBuild(true);
    try {
      const headers = getAuthHeaders();
      const res = await fetch("/api/tools/build-squad", {
        method: "POST",
        headers: { ...headers, "content-type": "application/json" },
        body: JSON.stringify({ budget: 1000 }),
      });
      const data = await res.json();
      if (data?.squad) {
        const wrapped = wrapSquadIfNeeded(data.squad);
        setToolSquad(wrapped);
        toastManager.success("Squad built successfully!");
      } else {
        toastManager.error("Failed to build squad: " + (data?.error || "Unknown error"));
      }
    } catch (e) {
      console.warn("Build squad failed:", e);
      toastManager.error("Failed to build squad");
    } finally {
      setLoadingBuild(false);
    }
  };

  const onSuggestTransfers = async () => {
    setLoadingSuggest(true);
    try {
      const headers = getAuthHeaders();
      const res = await fetch("/api/tools/suggest-transfers", {
        method: "POST",
        headers: { ...headers, "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (data?.transfers) {
        setToolTransfers(data.transfers);
        toastManager.success(`Found ${data.transfers.length} transfer suggestions`);
      } else {
        toastManager.error("Failed to get transfer suggestions: " + (data?.error || "Unknown error"));
      }
    } catch (e) {
      console.warn("Suggest transfers failed:", e);
      toastManager.error("Failed to get transfer suggestions");
    } finally {
      setLoadingSuggest(false);
    }
  };

  const onLoadSquad = (squad: any) => {
    const wrapped = wrapSquadIfNeeded(squad);
    setToolSquad(wrapped);
    toastManager.success("Squad loaded successfully!");
  };

  const onSaveCurrentSquad = () => {
    toastManager.success("Squad saved successfully!");
  };

  return (
    <div className="p-6 space-y-4">
      <ToastContainer />
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">FPL Assistant</h1>
          <p className="text-muted-foreground text-sm">
            Chat-powered squad building with persistence • Autosaves your progress
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="border rounded-md p-3 h-[70vh] lg:col-span-2">
          <AssistantRuntimeProvider runtime={runtime}>
            <Thread />
          </AssistantRuntimeProvider>
        </div>

        <div className="flex flex-col gap-4">
          {/* Quick actions */}
          <div className="border rounded-md p-3 bg-card flex items-center gap-2">
            <Button size="sm" onClick={onBuildSquad} disabled={loadingBuild}>
              {loadingBuild ? "Building..." : "Build Squad"}
            </Button>
            <Button size="sm" variant="outline" onClick={onSuggestTransfers} disabled={loadingSuggest}>
              {loadingSuggest ? "Suggesting..." : "Suggest Transfers"}
            </Button>
          </div>

          {/* Saved squads panel */}
          <SavedSquadsPanel 
            currentSquad={toolSquad}
            onLoadSquad={onLoadSquad}
            onSaveCurrentSquad={onSaveCurrentSquad}
          />

          {/* Squad panel: show chat result when available */}
          <SquadPanel toolSquad={toolSquad ?? undefined} header={toolSquad ? "Chat Squad" : "Wildcard Squad"} />

          {/* Transfer suggester */}
          <TransferSuggester />

          {/* Tool-based transfer suggestions */}
          {toolTransfers && (
            <div className="border rounded-md p-3 bg-card">
              <h2 className="text-lg font-semibold mb-2">Tool Suggestions</h2>
              {toolTransfers.length === 0 ? (
                <div className="text-sm text-muted-foreground">No suggestions.</div>
              ) : (
                <div className="grid gap-2 text-sm">
                  {toolTransfers.map((t: any, i: number) => (
                    <div key={i} className="rounded-md border bg-background/50 p-2">
                      <div className="font-medium">In: {t?.in?.name ?? t?.in?.id}</div>
                      {t?.reason && <div className="text-muted-foreground">{t.reason}</div>}
                      {typeof t?.estDeltaPoints === 'number' && (
                        <div className="text-xs">Δ points: {t.estDeltaPoints.toFixed(2)}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Explain picks: show selected squad details when available */}
          <ExplainPicks squad={toolSquad} />
        </div>
      </div>
    </div>
  );
}
