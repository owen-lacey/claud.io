"use client";

import React from 'react';
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from '@assistant-ui/react-ai-sdk';
import { Thread } from "./assistant-ui/thread";

const Chat: React.FC = () => {
  // Create a simple local runtime that connects to your API
  
  const runtime = useChatRuntime({ api: "/api/chat" });

  return (
    <div className="h-full flex flex-col">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Basic Chatbot</h3>
        <p className="text-sm text-gray-600">Test your API key with a simple chatbot</p>
      </div>
      
      <div className="flex-1 min-h-0">
        <AssistantRuntimeProvider runtime={runtime}>
          <Thread />
        </AssistantRuntimeProvider>
      </div>
    </div>
  );
};

export default Chat;
