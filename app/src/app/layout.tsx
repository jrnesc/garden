import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "garden",
  description: "An image becomes a place you can walk into.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">
        <svg aria-hidden="true" className="absolute h-0 w-0">
          <filter id="grain">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.75"
              numOctaves="2"
              seed="5"
              stitchTiles="stitch"
              result="noise"
            />
            <feDiffuseLighting
              in="noise"
              surfaceScale="2.5"
              diffuseConstant="1.2"
              lightingColor="#ffffff"
              result="bumps"
            >
              <feDistantLight azimuth="135" elevation="55" />
            </feDiffuseLighting>
            <feColorMatrix in="bumps" type="saturate" values="0" />
          </filter>
        </svg>
        {children}
      </body>
    </html>
  );
}
