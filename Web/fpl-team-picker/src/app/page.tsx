"use client";

import React, { useCallback, useEffect, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { ToastContainer, toastManager } from "@/components/ui/toast";
import { persistenceService } from "@/lib/persistence";
import AuthGuard from "@/components/AuthGuard";
import Header from "@/components/Header";
import SmallScreen from "@/components/utils/SmallScreen";
import { FplApi } from '@/helpers/fpl-api';
import { AllData } from '@/models/all-data';
import { ApiResult } from '@/models/api-result';
import { MyTeam as ApiMyTeam, League, Player, Team, User } from '@/helpers/api';
import { DataContext, RivalTeamsContext } from '@/lib/contexts';
import { RivalTeam } from '@/models/rival-league';
import { dataCache } from '@/lib/data-cache';

export default function HomePage() {
  const api = "/api/chat-tools";
  const [toolSquad, setToolSquad] = useState<any | null>(null);
  const [toolTransfers, setToolTransfers] = useState<any[] | null>(null);
  
  // Authentication and data loading
  const [plProfile, setPlProfile] = useState<string | null>(null);
  const [data, setData] = useState<AllData | null>(null);
  const [rivalTeams, setRivalTeams] = useState<RivalTeam[]>([]);

  const savePlProfile = (value: string | null) => {
    setPlProfile(value);
    if (typeof window !== 'undefined') {
      if (value) {
        localStorage.setItem("pl_profile", value);
      } else {
        localStorage.removeItem("pl_profile");
      }
    }
  };

  const loadData = useCallback(async () => {
    const api = new FplApi(plProfile || undefined);
    const [myTeam, players, teams, leagues, myDetails] = await Promise.all([
      api.myTeam.myTeamList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<ApiMyTeam>(false, err)),
      api.players.playersList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<Player[]>(false, err)),
      api.teams.teamsList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<Team[]>(false, err)),
      api.myLeagues.myLeaguesList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<League[]>(false, err)),
      api.myDetails.myDetailsList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<User>(false, err)),
    ]);
    const dataToSet = new AllData(myTeam, players, teams, leagues, myDetails);
    setData(dataToSet);
  }, [plProfile]);

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem("pl_profile");
      if (stored) {
        setPlProfile(stored);
      }
    }
  }, []);

  useEffect(() => {
    if (plProfile != null) {
      loadData();
    }
  }, [plProfile, loadData]);

  // Update global cache token when plProfile changes
  useEffect(() => {
    dataCache.updateToken(plProfile || undefined);
  }, [plProfile]);

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

  // Render logic
  if (!plProfile) {
    return (
      <DataContext.Provider value={data}>
        <RivalTeamsContext.Provider value={rivalTeams}>
          <div className="md:hidden">
            <SmallScreen />
          </div>
          <div className="hidden md:block">
            <AuthGuard onDone={(cookie) => savePlProfile(cookie)} />
          </div>
        </RivalTeamsContext.Provider>
      </DataContext.Provider>
    );
  }

  return (
    <DataContext.Provider value={data}>
      <RivalTeamsContext.Provider value={rivalTeams}>
        <div className="md:hidden">
          <SmallScreen />
        </div>
        <div className="hidden md:block">
          <div className="p-8 min-h-screen bg-background">
            <ToastContainer />
            
            {/* Header with user info */}
            <Header />

            {/* Full-width chat window */}
            <div className="mt-6">
              <div className="border border-border/50 rounded-xl p-4 h-[80vh] bg-card">
                <AssistantRuntimeProvider runtime={runtime}>
                  <Thread />
                </AssistantRuntimeProvider>
              </div>
            </div>
          </div>
        </div>
      </RivalTeamsContext.Provider>
    </DataContext.Provider>
  );
}
