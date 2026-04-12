# Board Image Sources

These are the best public candidates I found quickly for the two base-game map sides.
None of them is a clean orthographic scan, so expect some perspective correction or manual
coordinate tuning if you use them directly.

## Germany

- Best public candidate so far: [Funkenschlag-detail2.jpg](https://commons.wikimedia.org/wiki/File:Funkenschlag-detail2.jpg)
  - Wikimedia Commons
  - Listed size: 2400 x 1801
  - Notes: Original German-edition table photo and the best Germany-side reference I found from a stable public source.

- Alternative candidate: [Power Grid on Flickr](https://www.flickr.com/photos/martcatnoc/3515990632/sizes/o/)
  - Flickr
  - Listed size: 1000 x 768
  - Notes: Lower resolution, but still useful as a backup reference.

## USA

- Best public candidate so far: [Power Grid board game.jpg](https://commons.wikimedia.org/wiki/File:Power_Grid_board_game.jpg)
  - Wikimedia Commons
  - Listed size: 4032 x 3024
  - Notes: Highest-resolution public photo I found that clearly shows the board in play.

- Alternative candidate: [How Power Grid Works](https://entertainment.howstuffworks.com/leisure/brain-games/power-grid.htm)
  - HowStuffWorks article
  - Notes: The article includes a clear USA-side board photo, but the easily accessible web preview is much smaller than the Wikimedia candidate.

## Recommendation

If you want the fastest path forward:

1. Start with the Wikimedia Commons image for the side you want to calibrate first.
2. Save it locally as a PNG in `assets/boards/`.
3. Fill in coordinates in `board_layout_placeholders.json`.
4. Use `tests/manual_test/run_board_layout_preview.py` to tune the positions.

If you later find or create a flatter board scan, you can keep the same JSON schema and only retune the coordinates.
