"use client";

import {
  ComposerPrimitive,
  ThreadPrimitive,
  MessagePrimitive,
} from "@assistant-ui/react";
import type { FC } from "react";
import {
  MessageCircleIcon,
  SendHorizontalIcon,
  XIcon,
  CircleStopIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";

export const FloatingChat: FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {!isOpen && (
        <Button
          onClick={() => setIsOpen(true)}
          className="h-14 w-14 rounded-full shadow-lg"
          size="icon"
        >
          <MessageCircleIcon className="h-6 w-6" />
        </Button>
      )}
      
      {isOpen && (
        <ThreadPrimitive.Root className="bg-card border border-border shadow-xl rounded-lg w-96 h-[500px] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-border">
            <h3 className="text-lg font-semibold text-foreground">FPL Assistant</h3>
            <Button
              onClick={() => setIsOpen(false)}
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0"
            >
              <XIcon className="h-4 w-4" />
            </Button>
          </div>

          {/* Chat Messages Area */}
          <div className="flex-1 overflow-hidden">
            <ThreadPrimitive.Viewport className="h-full overflow-y-auto p-4 space-y-4">
              <ThreadPrimitive.Empty>
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <MessageCircleIcon className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-foreground font-medium mb-2">
                    Ask me anything about FPL!
                  </p>
                  <p className="text-muted-foreground text-sm">
                    I can help with team analysis, player recommendations, and strategy.
                  </p>
                </div>
              </ThreadPrimitive.Empty>

              <ThreadPrimitive.Messages
                components={{
                  UserMessage: FloatingUserMessage,
                  AssistantMessage: FloatingAssistantMessage,
                }}
              />
            </ThreadPrimitive.Viewport>
          </div>

          {/* Chat Input */}
          <div className="p-4 border-t border-border">
            <FloatingComposer />
          </div>
        </ThreadPrimitive.Root>
      )}
    </div>
  );
};

const FloatingComposer: FC = () => {
  return (
    <ComposerPrimitive.Root className="focus-within:border-ring/20 flex w-full items-end rounded-lg border bg-background px-2.5 shadow-sm transition-colors ease-in">
      <ComposerPrimitive.Input
        rows={1}
        placeholder="Ask about FPL strategy..."
        className="placeholder:text-muted-foreground max-h-32 flex-grow resize-none border-none bg-transparent px-2 py-3 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
      />
      <FloatingComposerAction />
    </ComposerPrimitive.Root>
  );
};

const FloatingComposerAction: FC = () => {
  return (
    <>
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send asChild>
          <TooltipIconButton
            tooltip="Send"
            variant="default"
            className="my-2.5 size-8 p-2 transition-opacity ease-in"
          >
            <SendHorizontalIcon />
          </TooltipIconButton>
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel asChild>
          <TooltipIconButton
            tooltip="Cancel"
            variant="default"
            className="my-2.5 size-8 p-2 transition-opacity ease-in"
          >
            <CircleStopIcon />
          </TooltipIconButton>
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>
    </>
  );
};

const FloatingUserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="flex justify-end mb-4">
      <div className="bg-primary text-primary-foreground max-w-[80%] break-words rounded-2xl rounded-br-sm px-4 py-2">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
};

const FloatingAssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="flex justify-start mb-4">
      <div className="bg-muted text-foreground max-w-[80%] break-words rounded-2xl rounded-bl-sm px-4 py-2">
        <MessagePrimitive.Parts components={{ Text: MarkdownText }} />
      </div>
    </MessagePrimitive.Root>
  );
};
