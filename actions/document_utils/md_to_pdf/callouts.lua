-- Pandoc Lua filter: render callouts ("encarts") as coloured LaTeX boxes.
--
-- Recognised syntaxes:
--   GitHub / Obsidian alerts      > [!NOTE] Optional title
--                                 > Body…
--   Pandoc fenced divs            ::: warning
--                                 Body…
--                                 :::
-- Types: note, info, tip, important, warning, caution (+ a few aliases such as
-- warn, danger, hint). A blockquote may chain several alerts: each paragraph
-- starting with [!TYPE] opens a new box. The boxes (tcolorbox "callout",
-- defined in md_to_pdf.py's LaTeX header) break across pages.

if not FORMAT:match('latex') then
  return {}
end

-- type → { colour name defined in the LaTeX header, default title }
local TYPES = {
  note      = { 'callout-note', 'Note' },
  info      = { 'callout-note', 'Info' },
  tip       = { 'callout-tip', 'Astuce' },
  hint      = { 'callout-tip', 'Astuce' },
  success   = { 'callout-tip', 'Succès' },
  important = { 'callout-important', 'Important' },
  question  = { 'callout-important', 'Question' },
  warning   = { 'callout-warning', 'Attention' },
  warn      = { 'callout-warning', 'Attention' },
  caution   = { 'callout-caution', 'Prudence' },
  danger    = { 'callout-caution', 'Danger' },
  error     = { 'callout-caution', 'Erreur' },
}

local function inlines_latex(inlines)
  local latex = pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), 'latex')
  return (latex:gsub('%s+$', ''))
end

local function box(kind, title_inlines, blocks)
  local spec = TYPES[kind]
  local title = (title_inlines and #title_inlines > 0)
    and inlines_latex(title_inlines) or spec[2]
  local out = { pandoc.RawBlock('latex',
    string.format('\\begin{callout}{%s}{%s}', spec[1], title)) }
  for _, b in ipairs(blocks) do out[#out + 1] = b end
  out[#out + 1] = pandoc.RawBlock('latex', '\\end{callout}')
  return out
end

-- "[!NOTE]", "[!note]-", "[!TIP]+" → "note" (only known types).
local function marker_type(inl)
  if not inl or inl.t ~= 'Str' then return nil end
  local kind = inl.text:match('^%[!(%a+)%][+-]?$')
  kind = kind and kind:lower()
  return TYPES[kind] and kind or nil
end

local function is_break(inl)
  return inl.t == 'LineBreak' or inl.t == 'SoftBreak'
end

-- Split a paragraph that starts with a marker into (kind, title, rest-para).
local function parse_marker(block)
  if block.t ~= 'Para' and block.t ~= 'Plain' then return nil end
  local kind = marker_type(block.content[1])
  if not kind then return nil end
  local title, rest = pandoc.List(), pandoc.List()
  local i, n = 2, #block.content
  while i <= n and block.content[i].t == 'Space' do i = i + 1 end
  while i <= n and not is_break(block.content[i]) do
    title:insert(block.content[i]); i = i + 1
  end
  for j = i + 1, n do rest:insert(block.content[j]) end
  while #title > 0 and title[#title].t == 'Space' do title:remove() end
  local body = #rest > 0 and pandoc.Para(rest) or nil
  return kind, title, body
end

function BlockQuote(bq)
  if not parse_marker(bq.content[1] or pandoc.Null()) then return nil end
  local out, cur = {}, nil
  local function flush()
    if cur then
      for _, b in ipairs(box(cur.kind, cur.title, cur.blocks)) do out[#out + 1] = b end
    end
  end
  for _, block in ipairs(bq.content) do
    local kind, title, body = parse_marker(block)
    if kind then
      flush()
      cur = { kind = kind, title = title, blocks = {} }
      if body then cur.blocks[#cur.blocks + 1] = body end
    else
      cur.blocks[#cur.blocks + 1] = block
    end
  end
  flush()
  return out
end

function Div(div)
  for _, cls in ipairs(div.classes) do
    local kind = cls:lower()
    if TYPES[kind] then
      local title = div.attributes.title
      return box(kind, title and pandoc.Inlines(title) or nil, div.content)
    end
  end
  return nil
end
