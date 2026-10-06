import torch

def solve(
    grid: torch.Tensor,
    result: torch.Tensor,
    rows: int,
    cols: int,
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
):
    g = grid.view(rows, cols)
    result.fill_(-1)

    if bool(g[start_row, start_col]) or bool(g[end_row, end_col]):
        return

    if start_row == end_row and start_col == end_col:
        result.fill_(0)
        return

    visited = torch.zeros(
        (rows, cols),
        dtype=torch.bool,
        device=grid.device,
    )
    visited[start_row, start_col] = True

    frontier = torch.zeros_like(visited)
    frontier[start_row, start_col] = True
    
    free = (g == 0)

    distance = 0
    while bool(frontier.any()):
        next_frontier = torch.zeros_like(frontier)

        # down
        next_frontier[1:, :] |= frontier[:-1, :]
        # up
        next_frontier[:-1, :] |= frontier[1:, :]
        # right
        next_frontier[:, 1:] |= frontier[:, :-1]
        # left
        next_frontier[:, :-1] |= frontier[:, 1:]

        next_frontier &= free
        next_frontier &= ~visited
        distance += 1

        if bool(next_frontier[end_row, end_col]):
            result.fill_(distance)
            return

        visited |= next_frontier
        frontier = next_frontier