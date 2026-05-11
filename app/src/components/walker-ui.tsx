"use client";

import type { FormEvent } from "react";
import Link from "next/link";
import {
  IconArrowLeft,
  IconBrush,
  IconCamera,
  IconClose,
  IconGrid,
  IconKeyboard,
  IconTrash,
  IconUndo,
  HUD_FONT,
  HUD_BOX_BASE,
  HUD_BOX_SQUARE,
} from "./hud-icons";

export type WalkerMenuTab = "edit" | "game" | "controls";

type CharacterOption = {
  label: string;
  glb: string;
  onnx: string;
  data: string;
  defaultStyle: string;
};

type SpeedOption = {
  label: string;
  value: number;
};

type WalkerHudProps = {
  backHref: string;
  cameraFollowUI: boolean;
  chaseCaught: boolean;
  chaseCount: number;
  chaseMode: boolean;
  chaseSurvived: number;
  deliveryDelivered: number;
  deliveryMode: boolean;
  menuOpen: boolean;
  paintMode: boolean;
  rampSelectedUI: boolean;
  splatLoading: boolean;
  viewMode: 0 | 1 | 2;
  onClear: () => void;
  onCycleViewMode: () => void;
  onOpenControls: () => void;
  onOpenEdit: () => void;
  onToggleCameraFollow: () => void;
  onUndo: () => void;
};

type WalkerMenuProps = {
  activeChar: string;
  activeSpeed: string;
  activeStyle: string;
  characters: CharacterOption[];
  charLoading: boolean;
  chaseMode: boolean;
  colorCss: Record<string, string>;
  deliveryMode: boolean;
  error: string | null;
  intent: string;
  lastEdit: { reason: string | null; usage: Record<string, unknown> | null } | null;
  menuOpen: boolean;
  menuTab: WalkerMenuTab;
  paintColor: string;
  paintMode: boolean;
  paintOp: string;
  paintSize: string;
  sizeMap: Record<string, number>;
  speedTiers: SpeedOption[];
  styles: string[];
  submitting: boolean;
  onBackdropClose: () => void;
  onClose: () => void;
  onIntentChange: (value: string) => void;
  onIntentSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onLoadCharacter: (character: CharacterOption) => void;
  onPickSpeed: (label: string, value: number) => void;
  onPickStyle: (style: string) => void;
  onSetChaseMode: (value: boolean | ((enabled: boolean) => boolean)) => void;
  onSetDeliveryMode: (value: boolean | ((enabled: boolean) => boolean)) => void;
  onSetMenuOpen: (open: boolean) => void;
  onSetMenuTab: (tab: WalkerMenuTab) => void;
  onSetPaintColor: (color: string) => void;
  onSetPaintMode: (value: boolean | ((enabled: boolean) => boolean)) => void;
  onSetPaintOp: (op: string) => void;
  onSetPaintSize: (size: string) => void;
};

export function WalkerHud({
  backHref,
  cameraFollowUI,
  chaseCaught,
  chaseCount,
  chaseMode,
  chaseSurvived,
  deliveryDelivered,
  deliveryMode,
  menuOpen,
  paintMode,
  rampSelectedUI,
  splatLoading,
  viewMode,
  onClear,
  onCycleViewMode,
  onOpenControls,
  onOpenEdit,
  onToggleCameraFollow,
  onUndo,
}: WalkerHudProps) {
  if (menuOpen || splatLoading) return null;

  return (
    <div style={HUD_FONT} className="pointer-events-none">
      <Link
        href={backHref}
        aria-label="back"
        className={`pointer-events-auto absolute left-5 top-5 z-10 ${HUD_BOX_SQUARE}`}
      >
        <IconArrowLeft />
      </Link>

      {deliveryMode && (
        <div className="absolute right-5 top-5 z-10">
          <div className="rounded-2xl border border-white/10 bg-black/55 px-4 py-3 text-right backdrop-blur-md">
            <div className="text-[10px] uppercase tracking-[0.16em] text-zinc-400">Delivered</div>
            <div className="mt-1 text-[22px] leading-none text-white">
              {deliveryDelivered}/3
            </div>
            <div
              className={`mt-2 h-1.5 w-28 overflow-hidden rounded-full bg-white/10 ${
                deliveryDelivered === 3 ? "ring-1 ring-emerald-300/50" : ""
              }`}
            >
              <div
                className="h-full rounded-full bg-emerald-300 transition-all duration-300"
                style={{ width: `${(deliveryDelivered / 3) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {deliveryMode && rampSelectedUI && (
        <div className="absolute left-5 top-1/2 z-10 -translate-y-1/2">
          <div className="rounded-2xl border border-cyan-200/20 bg-black/55 px-4 py-3 backdrop-blur-md">
            <div className="text-[10px] uppercase tracking-[0.16em] text-cyan-200">Ramp</div>
            <div className="mt-1 text-[13px] text-white">Click floor to place</div>
          </div>
        </div>
      )}

      {deliveryMode && deliveryDelivered === 3 && (
        <div className="absolute left-1/2 top-[88px] z-10 -translate-x-1/2">
          <div className="rounded-2xl border border-emerald-200/25 bg-black/60 px-5 py-3 text-center backdrop-blur-md">
            <div className="text-[10px] uppercase tracking-[0.18em] text-emerald-200">Complete</div>
            <div className="mt-1 text-[14px] text-white">All artifacts delivered</div>
          </div>
        </div>
      )}

      {chaseMode && (
        <div className="absolute right-5 top-5 z-10">
          <div className="rounded-2xl border border-white/10 bg-black/55 px-4 py-3 text-right backdrop-blur-md">
            <div className="text-[10px] uppercase tracking-[0.16em] text-zinc-400">
              {chaseCaught ? "Caught" : "Survive"}
            </div>
            <div className="mt-1 text-[22px] leading-none text-white">
              {chaseSurvived.toFixed(1)}s
            </div>
            <div className="mt-2 text-[12px] text-zinc-300">
              {chaseCount} zombies
            </div>
          </div>
        </div>
      )}

      <div className="pointer-events-auto absolute bottom-6 left-1/2 z-10 -translate-x-1/2">
        <div className="flex items-center gap-1 rounded-2xl border border-white/10 bg-black/55 p-1.5 backdrop-blur-md">
          <button
            onClick={onUndo}
            aria-label="undo"
            title="undo (z)"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-300 hover:bg-white/10 hover:text-white"
          >
            <IconUndo />
          </button>
          <button
            onClick={onClear}
            aria-label="clear edits"
            title="clear all (shift+z)"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-300 hover:bg-white/10 hover:text-white"
          >
            <IconTrash />
          </button>
          <div className="mx-1 h-5 w-px bg-white/10" />
          <button
            onClick={onOpenEdit}
            aria-label="edit"
            title="edit world"
            className={`flex h-9 w-9 items-center justify-center rounded-lg ${
              paintMode ? "bg-white/15 text-white" : "text-zinc-300 hover:bg-white/10 hover:text-white"
            }`}
          >
            <IconBrush />
          </button>
          <button
            onClick={onCycleViewMode}
            aria-label="view mode"
            title={
              viewMode === 0
                ? "view: mesh - press m for splat+mesh"
                : viewMode === 1
                  ? "view: splat + mesh - press m for splat"
                  : "view: splat - press m for mesh"
            }
            className={`flex h-9 w-9 items-center justify-center rounded-lg ${
              viewMode !== 0 ? "bg-white/15 text-white" : "text-zinc-300 hover:bg-white/10 hover:text-white"
            }`}
          >
            <IconGrid />
          </button>
          <button
            onClick={onToggleCameraFollow}
            aria-label="camera follow"
            title={cameraFollowUI ? "camera: locked" : "camera: free"}
            className={`flex h-9 w-9 items-center justify-center rounded-lg ${
              cameraFollowUI ? "bg-white/15 text-white" : "text-zinc-300 hover:bg-white/10 hover:text-white"
            }`}
          >
            <IconCamera />
          </button>
          <button
            onClick={onOpenControls}
            aria-label="controls"
            title="controls"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-300 hover:bg-white/10 hover:text-white"
          >
            <IconKeyboard />
          </button>
        </div>
      </div>
    </div>
  );
}

export function WalkerMenu({
  activeChar,
  activeSpeed,
  activeStyle,
  characters,
  charLoading,
  chaseMode,
  colorCss,
  deliveryMode,
  error,
  intent,
  lastEdit,
  menuOpen,
  menuTab,
  paintColor,
  paintMode,
  paintOp,
  paintSize,
  sizeMap,
  speedTiers,
  styles,
  submitting,
  onBackdropClose,
  onClose,
  onIntentChange,
  onIntentSubmit,
  onLoadCharacter,
  onPickSpeed,
  onPickStyle,
  onSetChaseMode,
  onSetDeliveryMode,
  onSetMenuOpen,
  onSetMenuTab,
  onSetPaintColor,
  onSetPaintMode,
  onSetPaintOp,
  onSetPaintSize,
}: WalkerMenuProps) {
  if (!menuOpen) return null;

  return (
    <div
      style={HUD_FONT}
      className="absolute inset-0 z-20 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onBackdropClose();
      }}
    >
      <div className="h-[560px] w-[500px] overflow-hidden rounded-2xl border border-white/10 bg-zinc-950/95 shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-2 py-2">
          <div className="flex items-center gap-1">
            {(["edit", "game", "controls"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => onSetMenuTab(tab)}
                className={`rounded-md px-3 py-1.5 text-[13px] ${
                  menuTab === tab ? "bg-white/10 text-white" : "text-zinc-400 hover:text-white"
                }`}
              >
                {tab[0].toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          <button
            onClick={onClose}
            aria-label="close"
            className="flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 hover:bg-white/10 hover:text-white"
          >
            <IconClose />
          </button>
        </div>

        <div className="h-[507px] p-5">
          {menuTab === "edit" && (
            <div className="space-y-5">
              <button
                onClick={() => {
                  onSetPaintMode((enabled) => !enabled);
                  onSetMenuOpen(false);
                }}
                className={`w-full rounded-lg border px-4 py-2.5 text-[13px] transition ${
                  paintMode
                    ? "border-white/30 bg-white/15 text-white"
                    : "border-white/10 bg-white/5 text-zinc-400 hover:border-white/20 hover:text-white"
                }`}
              >
                {paintMode ? "Painting - click & drag on world" : "Enable paintbrush"}
              </button>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Color</div>
                <div className="flex flex-wrap gap-2">
                  {Object.keys(colorCss).map((color) => (
                    <button
                      key={color}
                      onClick={() => {
                        onSetPaintColor(color);
                        onSetPaintOp("recolor");
                      }}
                      title={color}
                      className={`h-7 w-7 rounded-full border-2 transition ${
                        paintColor === color && paintOp === "recolor"
                          ? "border-white scale-110"
                          : "border-transparent hover:border-white/40"
                      }`}
                      style={{ backgroundColor: colorCss[color] }}
                    />
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Tool</div>
                <div className="flex gap-2">
                  {(["recolor", "erase", "brighten", "darken"] as const).map((op) => (
                    <button
                      key={op}
                      onClick={() => onSetPaintOp(op)}
                      className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                        paintOp === op ? "border-white/30 bg-white/10 text-white" : ""
                      }`}
                    >
                      {op}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Size</div>
                <div className="flex gap-2">
                  {Object.keys(sizeMap).map((size) => (
                    <button
                      key={size}
                      onClick={() => onSetPaintSize(size)}
                      className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                        paintSize === size ? "border-white/30 bg-white/10 text-white" : ""
                      }`}
                    >
                      {size}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">AI Edit</div>
                <form onSubmit={onIntentSubmit}>
                  <input
                    type="text"
                    value={intent}
                    onChange={(e) => onIntentChange(e.target.value)}
                    placeholder={submitting ? "thinking..." : "make the table blue"}
                    disabled={submitting}
                    className="w-full rounded-lg border border-white/10 bg-white/5 px-3.5 py-2.5 text-[13px] text-white placeholder:text-zinc-500 focus:border-white/30 focus:outline-none disabled:opacity-50"
                  />
                  {error && menuTab === "edit" && (
                    <div className="mt-3 text-[12px] text-red-400">{error}</div>
                  )}
                  {lastEdit?.reason && !submitting && !error && (
                    <div className="mt-3 text-[12px] leading-relaxed text-zinc-400">
                      {lastEdit.reason}
                    </div>
                  )}
                </form>
              </div>
            </div>
          )}

          {menuTab === "game" && (
            <div className="space-y-3">
              <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Games</div>
              <button
                onClick={() => {
                  onSetDeliveryMode((enabled) => !enabled);
                  onSetChaseMode(false);
                }}
                className={`flex w-full items-center justify-between rounded-lg border px-4 py-3 text-left transition ${
                  deliveryMode
                    ? "border-emerald-200/30 bg-emerald-300/15 text-emerald-100"
                    : "border-white/10 bg-white/5 text-zinc-300 hover:border-white/20 hover:text-white"
                }`}
              >
                <span className="text-[14px] text-white">Artifact Delivery</span>
                <span className="text-[12px]">{deliveryMode ? "On" : "Off"}</span>
              </button>
              <button
                onClick={() => {
                  onSetChaseMode((enabled) => !enabled);
                  onSetDeliveryMode(false);
                }}
                className={`flex w-full items-center justify-between rounded-lg border px-4 py-3 text-left transition ${
                  chaseMode
                    ? "border-pink-200/30 bg-pink-300/15 text-pink-100"
                    : "border-white/10 bg-white/5 text-zinc-300 hover:border-white/20 hover:text-white"
                }`}
              >
                <span className="text-[14px] text-white">Zombie Chase</span>
                <span className="text-[12px]">{chaseMode ? "On" : "Off"}</span>
              </button>
            </div>
          )}

          {menuTab === "controls" && (
            <div className="space-y-4">
              <div className="space-y-2 text-[13px]">
                <div className="flex justify-between">
                  <span className="text-white">Walk</span>
                  <span className="text-zinc-400">wasd</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white">Orbit camera</span>
                  <span className="text-zinc-400">click + drag</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white">Zoom</span>
                  <span className="text-zinc-400">scroll</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white">Undo / clear</span>
                  <span className="text-zinc-400">z / shift+z</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white">Mesh overlay</span>
                  <span className="text-zinc-400">m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white">Game options</span>
                  <span className="text-zinc-400">Game tab</span>
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Character</div>
                <div className="flex gap-2">
                  {characters.map((character) => (
                    <button
                      key={character.label}
                      onClick={() => onLoadCharacter(character)}
                      disabled={charLoading}
                      className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                        activeChar === character.label ? "border-white/30 bg-white/10 text-white" : ""
                      }`}
                    >
                      {character.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Speed</div>
                <div className="flex gap-2">
                  {speedTiers.map((tier) => (
                    <button
                      key={tier.label}
                      onClick={() => onPickSpeed(tier.label, tier.value)}
                      className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                        activeSpeed === tier.label ? "border-white/30 bg-white/10 text-white" : ""
                      }`}
                    >
                      {tier.label}
                    </button>
                  ))}
                </div>
              </div>

              {styles.length > 0 && (
                <div>
                  <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Style</div>
                  <div className="flex flex-wrap gap-2">
                    {styles.map((style) => (
                      <button
                        key={style}
                        onClick={() => onPickStyle(style)}
                        className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                          activeStyle === style ? "border-white/30 bg-white/10 text-white" : ""
                        }`}
                      >
                        {style}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
