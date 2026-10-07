# Hosting the web demo

`web/index.html` is a single self-contained page (HTML, CSS and JavaScript inline; Plotly and the
fonts load from public CDNs). It has no build step and no server code, so it works from any path.

## On nitipong.com (Next.js on Vercel)

1. In the website repository, copy the page:

   ```
   public/phasespace-distance/index.html   <-  web/index.html
   ```

2. Next.js serves `public/phasespace-distance/index.html` at `/phasespace-distance/index.html`.
   To make the short address `nitipong.com/phasespace-distance` work, add a rewrite to `next.config.js`
   (or `next.config.mjs`):

   ```js
   const nextConfig = {
     async rewrites() {
       return [
         { source: "/phasespace-distance", destination: "/phasespace-distance/index.html" },
         { source: "/phasespace-distance/", destination: "/phasespace-distance/index.html" },
       ];
     },
   };
   module.exports = nextConfig;   // or: export default nextConfig;
   ```

   If the site uses `output: "export"` (static export, where rewrites are not available), set
   `trailingSlash: true` instead; `/phasespace-distance/` then resolves to the folder's `index.html`.
   A `vercel.json` rewrite also works:

   ```json
   { "rewrites": [ { "source": "/phasespace-distance", "destination": "/phasespace-distance/index.html" } ] }
   ```

3. Commit and push; Vercel redeploys. Check `https://nitipong.com/phasespace-distance`.

To update the demo later, copy the new `web/index.html` over the old one.

## On GitHub Pages

The workflow `.github/workflows/pages.yml` publishes `web/` whenever it changes on `main`.
Enable it once under **Settings > Pages > Source: GitHub Actions**. The demo then appears at
`https://cnnitipong.github.io/phasespace-distance/`.

## Locally

Open `web/index.html` in a browser (an internet connection is needed for Plotly and the fonts), or run
`python -m http.server -d web` and visit `http://localhost:8000`.
