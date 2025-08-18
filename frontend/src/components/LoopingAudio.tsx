import React, { useEffect, useRef, useState } from "react";

export default function LoopingAudio({ defaultUrl = "" }: { defaultUrl?: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [src, setSrc] = useState<string>(defaultUrl);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.35);

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
  }, [volume]);

  useEffect(() => {
    // stop playback if source changes
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsPlaying(false);
  }, [src]);

  async function toggle() {
    const a = audioRef.current;
    if (!a) return;
    if (isPlaying) {
      a.pause();
      setIsPlaying(false);
    } else {
      try {
        await a.play();          // user-gesture required → call from button
        setIsPlaying(true);
      } catch (e) {
        console.error(e);
        alert("Browser blocked audio autoplay. Click the button again after selecting a source.");
      }
    }
  }

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    if (src.startsWith("blob:")) URL.revokeObjectURL(src);
    setSrc(url);
  }

  return (
    <>
      <audio ref={audioRef} src={src || undefined} loop preload="auto" crossOrigin="anonymous" />
      <button type="button" onClick={toggle} disabled={!src}>
        {isPlaying ? "Pause Sound" : "Play Sound (Loop)"}
      </button>

      <input type="file" accept="audio/*" onChange={onPickFile} />
      <input
        type="text"
        placeholder="or paste audio URL (mp3/wav/ogg)"
        value={src}
        onChange={(e) => setSrc(e.target.value)}
        style={{ width: 260 }}
      />

      <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        Volume
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          onChange={(e) => setVolume(parseFloat(e.target.value))}
        />
      </label>
    </>
  );
}
