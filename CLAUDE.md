# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (required before first run)
pnpm install

# Development server with live reload
hugo server --disableFastRender

# Production build
hugo --minify

# Full production build (with search indexing, as used in CI)
hugo --gc --minify && pnpm dlx pagefind --source 'public'
```

**Requirements:** Hugo Extended 0.152.1, Node 20+, pnpm, Go 1.19+

## Architecture

Hugo Blox academic CV template. Content is pure Markdown with YAML frontmatter; no custom theme code — styling comes from `blox-tailwind` Hugo module (Tailwind CSS v4 + Preact).

**Config:** `config/_default/` — `hugo.yaml` (core), `params.yaml` (appearance/features), `menus.yaml` (nav), `module.yaml` (Hugo modules).

**Content structure:**
- `content/_index.md` — Homepage; uses Hugo Blox block types (`resume-biography-3`, `collection`, etc.) declared in frontmatter
- `content/experience.md` — CV page with `resume-experience` and `resume-skills` blocks
- `content/authors/admin/_index.md` — Author profile (name, bio, socials, education)
- `content/projects/*/index.md` — Project pages; each is a standalone article with optional images via `{{< figure src="..." >}}`
- `content/publications/` — Auto-generated from BibTeX via GitHub Actions (`import-publications.yml`)
- `static/uploads/` — Static assets (e.g., `CV.pdf`)

**Custom layouts:** Only `layouts/partials/hooks/head-end/github-button.html` (loads GitHub buttons JS). All other rendering comes from the Hugo Blox module.

## Content Authoring

**Adding a project:** Create `content/projects/<name>/index.md` with frontmatter including `title`, `date`, `tags`, `image`, and optional `links` array. Add images alongside the index file and reference them with `{{< figure src="filename.png" >}}`.

**Homepage blocks** are configured via the `sections` array in `content/_index.md` frontmatter — each block has a `block` type key and block-specific params.

**Math rendering** is enabled site-wide; use `$...$` for inline and `$$...$$` for display math.

## Deployment

GitHub Actions (`deploy.yml`) builds on push to `main` and deploys to GitHub Pages. Publications are auto-imported via `import-publications.yml` when `publications.bib` changes.
