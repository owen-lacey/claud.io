"use client";

import { useEffect, useState } from "react";
import { persistenceService, type PersistedSquad } from "@/lib/persistence";
import { Button } from "@/components/ui/button";
import { Trash2, Edit3, Download, Upload, FolderOpen } from "lucide-react";

export type SavedSquadsPanelProps = {
  currentSquad?: any | null;
  onLoadSquad?: (squad: any) => void;
  onSaveCurrentSquad?: () => void;
};

export default function SavedSquadsPanel({ 
  currentSquad, 
  onLoadSquad, 
  onSaveCurrentSquad 
}: SavedSquadsPanelProps) {
  const [savedSquads, setSavedSquads] = useState<PersistedSquad[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");

  const refreshSquads = () => {
    setSavedSquads(persistenceService.getSavedSquads());
  };

  useEffect(() => {
    refreshSquads();
  }, []);

  const handleSaveCurrent = () => {
    if (!currentSquad) return;
    persistenceService.saveSquad(currentSquad);
    refreshSquads();
    onSaveCurrentSquad?.();
  };

  const handleDelete = (id: string) => {
    persistenceService.deleteSquad(id);
    refreshSquads();
  };

  const handleStartEdit = (squad: PersistedSquad) => {
    setEditingId(squad.id);
    setEditingName(squad.name);
  };

  const handleSaveEdit = (id: string) => {
    if (editingName.trim()) {
      persistenceService.renameSquad(id, editingName.trim());
      refreshSquads();
    }
    setEditingId(null);
    setEditingName("");
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditingName("");
  };

  const handleLoad = (squad: PersistedSquad) => {
    onLoadSquad?.(squad.squad);
  };

  const handleExport = () => {
    const data = persistenceService.exportSession();
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fpl-assistant-session-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        persistenceService.importSession(content);
        refreshSquads();
        alert("Session imported successfully!");
      } catch (error) {
        alert("Failed to import session: " + (error as Error).message);
      }
    };
    reader.readAsText(file);
    event.target.value = ""; // Reset input
  };

  return (
    <div className="border rounded-md p-3 flex flex-col gap-3 bg-card">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Saved Squads</h2>
          <p className="text-xs text-muted-foreground">{savedSquads.length} saved</p>
        </div>
        <div className="flex items-center gap-1">
          <Button 
            size="sm" 
            variant="outline" 
            onClick={handleSaveCurrent}
            disabled={!currentSquad}
            title="Save current squad"
          >
            Save
          </Button>
          <Button 
            size="sm" 
            variant="ghost" 
            onClick={handleExport}
            title="Export all data"
          >
            <Download className="h-4 w-4" />
          </Button>
          <Button 
            size="sm" 
            variant="ghost" 
            onClick={() => document.getElementById("import-file")?.click()}
            title="Import data"
          >
            <Upload className="h-4 w-4" />
          </Button>
          <input
            id="import-file"
            type="file"
            accept=".json"
            onChange={handleImport}
            className="hidden"
          />
        </div>
      </div>

      {savedSquads.length === 0 ? (
        <div className="text-sm text-muted-foreground text-center py-4">
          No saved squads yet. Build a squad and save it!
        </div>
      ) : (
        <div className="flex flex-col gap-2 max-h-64 overflow-y-auto">
          {savedSquads.map((squad) => (
            <div key={squad.id} className="rounded-md border bg-background/50 p-2">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  {editingId === squad.id ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        className="flex-1 px-2 py-1 text-sm border rounded"
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleSaveEdit(squad.id);
                          if (e.key === "Escape") handleCancelEdit();
                        }}
                        autoFocus
                      />
                      <Button size="sm" onClick={() => handleSaveEdit(squad.id)}>Save</Button>
                      <Button size="sm" variant="ghost" onClick={handleCancelEdit}>Cancel</Button>
                    </div>
                  ) : (
                    <div>
                      <div className="font-medium text-sm truncate">{squad.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(squad.createdAt).toLocaleDateString()}
                      </div>
                    </div>
                  )}
                </div>
                {editingId !== squad.id && (
                  <div className="flex items-center gap-1 ml-2">
                    <Button 
                      size="sm" 
                      variant="ghost" 
                      onClick={() => handleLoad(squad)}
                      title="Load this squad"
                    >
                      <FolderOpen className="h-3 w-3" />
                    </Button>
                    <Button 
                      size="sm" 
                      variant="ghost" 
                      onClick={() => handleStartEdit(squad)}
                      title="Rename"
                    >
                      <Edit3 className="h-3 w-3" />
                    </Button>
                    <Button 
                      size="sm" 
                      variant="ghost" 
                      onClick={() => handleDelete(squad.id)}
                      title="Delete"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
