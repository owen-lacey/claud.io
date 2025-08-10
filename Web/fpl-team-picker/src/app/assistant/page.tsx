"use client";

import React, { useEffect, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";

export default function AssistantPage() {
  const api = "/api/chat-tools";
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
            <li>Ask: &quot;Call build_squad with budget 1000 and return only the result.&quot;</li>
            <li>Look for f: (tool call) and d: (tool result) frames in the Network tab.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
