-- Pandoc Lua filter: render tables as LaTeX longtables that stay readable.
--
-- Pandoc's default LaTeX tables take their column widths from the dashes of
-- the Markdown separator line (|---|---|), so a column of short labels can end
-- up narrower than its longest word and spill into its neighbour, and rows run
-- into each other with no separation. This filter instead:
--   * sizes each column from its content: never narrower than its longest
--     word, the remaining width shared out by amount of text;
--   * keeps natural-width columns when the whole table fits on one line;
--   * draws a light rule between body rows and sets wide tables in \small.
-- Tables with merged cells (row/col spans) are left to pandoc.

if not FORMAT:match('latex') then
  return {}
end

local stringify = pandoc.utils.stringify

-- Text width in characters of \small TeX Gyre Heros at 1in margins (approx.).
local LINE_CHARS = 93
-- Width taken by the inter-column padding (2\tabcolsep), in characters.
local PAD_CHARS = 2.4
local BOLD_FACTOR = 1.1

local function ulen(s)
  return utf8.len(s) or #s
end

local function has_strong(blocks)
  local found = false
  pandoc.walk_block(pandoc.Div(blocks), {
    Strong = function() found = true end,
  })
  return found
end

-- Natural width (whole cell on one line) and minimum width (longest word).
local function measure(cell)
  local text = stringify(cell.contents)
  local factor = has_strong(cell.contents) and BOLD_FACTOR or 1
  local longest = 0
  for word in text:gmatch('%S+') do
    longest = math.max(longest, ulen(word))
  end
  return ulen(text) * factor, longest * factor
end

local function all_rows(tbl)
  local rows = {}
  for _, r in ipairs(tbl.head.rows) do rows[#rows + 1] = r end
  for _, body in ipairs(tbl.bodies) do
    for _, r in ipairs(body.head) do rows[#rows + 1] = r end
    for _, r in ipairs(body.body) do rows[#rows + 1] = r end
  end
  for _, r in ipairs(tbl.foot.rows) do rows[#rows + 1] = r end
  return rows
end

local function has_spans(tbl)
  for _, row in ipairs(all_rows(tbl)) do
    for _, cell in ipairs(row.cells) do
      if cell.row_span > 1 or cell.col_span > 1 then return true end
    end
  end
  return false
end

-- Share `avail` characters between columns: proportional to amount of text,
-- but never below a column's minimum; columns pinned at their minimum are
-- removed and the rest redistributed until stable.
local function distribute(nat, min, avail)
  local n = #nat
  local w, fixed = {}, {}
  for _ = 1, n do
    local free, free_nat = avail, 0
    for i = 1, n do
      if fixed[i] then free = free - w[i] else free_nat = free_nat + nat[i] end
    end
    local changed = false
    for i = 1, n do
      if not fixed[i] then
        w[i] = free_nat > 0 and free * nat[i] / free_nat or free / n
        if w[i] < min[i] then
          w[i], fixed[i], changed = min[i], true, true
        end
      end
    end
    if not changed then break end
  end
  return w
end

local function cell_latex(cell)
  local latex = pandoc.write(pandoc.Pandoc(cell.contents), 'latex')
  -- Paragraph breaks inside a cell become explicit line breaks.
  latex = latex:gsub('%s+$', ''):gsub('\n\n+', '\\par ')
  return latex
end

local function row_latex(row)
  local parts = {}
  for _, cell in ipairs(row.cells) do parts[#parts + 1] = cell_latex(cell) end
  return table.concat(parts, ' & ') .. ' \\\\'
end

local ALIGN = {
  AlignLeft = { '\\raggedright', 'l' },
  AlignRight = { '\\raggedleft', 'r' },
  AlignCenter = { '\\centering', 'c' },
  AlignDefault = { '\\raggedright', 'l' },
}

local LIGHT_RULE = '\\arrayrulecolor{black!20}\\midrule\\arrayrulecolor{black}'

function Table(tbl)
  if has_spans(tbl) then return nil end

  local ncols = #tbl.colspecs
  local nat, min, total = {}, {}, {}
  for i = 1, ncols do nat[i], min[i], total[i] = 1, 1, 1 end
  for _, row in ipairs(all_rows(tbl)) do
    for i, cell in ipairs(row.cells) do
      local n, m = measure(cell)
      nat[i] = math.max(nat[i], n)
      min[i] = math.max(min[i], m)
      total[i] = total[i] + n
    end
  end

  local avail = LINE_CHARS - ncols * PAD_CHARS
  local total_nat = 0
  for i = 1, ncols do total_nat = total_nat + nat[i] end
  local fits = total_nat <= avail

  local spec = {}
  if fits then
    for i, cs in ipairs(tbl.colspecs) do spec[i] = ALIGN[cs[1]][2] end
  else
    -- Weight by total text so one long cell doesn't hog the width.
    local w = distribute(total, min, avail)
    local sum = 0
    for i = 1, ncols do sum = sum + w[i] end
    for i, cs in ipairs(tbl.colspecs) do
      spec[i] = string.format(
        '>{%s\\arraybackslash}p{(\\linewidth - %d\\tabcolsep) * \\real{%.4f}}',
        ALIGN[cs[1]][1], 2 * ncols, w[i] / sum)
    end
  end

  local out = {}
  local function add(s) out[#out + 1] = s end

  add('\\begingroup')
  if not fits then add('\\small') end
  add('\\renewcommand{\\arraystretch}{1.15}')
  add('\\begin{longtable}[]{@{}' .. table.concat(spec) .. '@{}}')

  local caption = stringify(tbl.caption.long or {})
  if caption ~= '' then
    add('\\caption{' .. pandoc.write(pandoc.Pandoc(tbl.caption.long), 'latex')
      :gsub('%s+$', '') .. '}\\tabularnewline')
  end

  local head = {}
  for _, r in ipairs(tbl.head.rows) do head[#head + 1] = row_latex(r) end
  -- Skip an all-empty header row (e.g. "| | |" headers).
  local empty_head = true
  for _, r in ipairs(tbl.head.rows) do
    for _, c in ipairs(r.cells) do
      if stringify(c.contents) ~= '' then empty_head = false end
    end
  end

  add('\\toprule\\noalign{}')
  if #head > 0 and not empty_head then
    add(table.concat(head, '\n'))
    add('\\midrule\\noalign{}')
    add('\\endhead')
  end
  add('\\bottomrule\\noalign{}')
  add('\\endlastfoot')

  local body = {}
  for _, b in ipairs(tbl.bodies) do
    for _, r in ipairs(b.head) do body[#body + 1] = row_latex(r) end
    for _, r in ipairs(b.body) do body[#body + 1] = row_latex(r) end
  end
  for _, r in ipairs(tbl.foot.rows) do body[#body + 1] = row_latex(r) end
  add(table.concat(body, '\n' .. LIGHT_RULE .. '\n'))

  add('\\end{longtable}')
  add('\\endgroup')

  return pandoc.RawBlock('latex', table.concat(out, '\n'))
end
