"use client";

import React, { useEffect } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";

export default function AssistantPage() {
  const api = "/api/chat-tools";

  // Attach auth headers for chat-tools calls
  useEffect(() => {
    const originalFetch = window.fetch;
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : "";
      if (url.startsWith(api)) {
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
          // Optional: also send as Bearer for servers that expect Authorization
          headers.set("Authorization", `Bearer ${token}`);
        }
        nextInit.headers = headers;
        return originalFetch(input, nextInit);
      }
      return originalFetch(input, init);
    };
    return () => {
      window.fetch = originalFetch;
    };
  }, [api]);

  const runtime = useChatRuntime({ api });

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Assistant</h1>
          <p className="text-muted-foreground text-sm">
            Minimal chat hooked to {api}.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="border rounded-md p-3 h-[70vh] lg:col-span-2">
          <AssistantRuntimeProvider runtime={runtime}>
            <Thread />
          </AssistantRuntimeProvider>
        </div>

        <div className="border rounded-md p-3 h-[70vh] overflow-auto text-xs bg-muted/30">
          <p className="font-medium mb-2">Debug: API endpoint</p>
          <pre className="text-muted-foreground">{api}</pre>
          <p className="font-medium mt-4 mb-2">Tip</p>
          <ul className="list-disc pl-5 text-muted-foreground">
            <li>Ask: &quot;Call build_squad and return only the result.&quot;</li>
            <li>Token attached from localStorage key pl_profile.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
