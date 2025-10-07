# Git Workflow Inside Neovim

This configuration ships with the Fugitive, Gitsigns, and Vimade plugins to cover the entire Git workflow without leaving Neovim.

## Status, Diffs, and Commits
- `:Git` opens Fugitive's status dashboard where you can stage, commit, and push interactively.
- Use `dv` inside the status buffer to open a vertical diff or `ds` for a horizontal diff.
- `:Gwrite` stages the current buffer, while `:Gread` resets it to HEAD.
- `:Gdiffsplit` compares the current buffer against the index; add `!` to compare against another branch or commit.

## Navigating History
- `:Glog` shows commit history scoped to the current file, with `<CR>` opening the selected revision in a split.
- Inside any Fugitive buffer, `~` toggles between staged and unstaged views.

## Inline Git Hints
- Gitsigns highlights added, removed, and modified hunks in the sign column.
- `[c` / `]c` jump between hunks; `:Gitsigns preview_hunk` shows the diff in a floating window.
- Use `<leader>hs` / `<leader>hr` to stage or reset the hunk under the cursor.

## Focused Reviewing
- Vimade automatically dims inactive windows so you can concentrate on the file that is currently under review.
- Toggle the dimming with `:VimadeToggleActive` when you need consistent brightness across splits.

## Helpful Links
- `:help fugitive.txt`
- `:help gitsigns.txt`
- `:help vimade`
