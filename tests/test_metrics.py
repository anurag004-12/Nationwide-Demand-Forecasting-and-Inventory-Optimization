from src.metrics import wape, rmse, coverage


def test_wape():
    # errors are 1 and 1 -> total error 2, total actual 10 -> 0.2
    assert abs(wape([4, 6], [5, 5]) - 0.2) < 1e-9


def test_rmse():
    # errors 3 and 4 -> sqrt((9 + 16) / 2)
    assert abs(rmse([0, 0], [3, 4]) - (12.5 ** 0.5)) < 1e-9


def test_coverage():
    # 3 of the 4 actual values fall inside their intervals
    assert coverage([1, 5, 9, 20], [0, 4, 8, 0], [2, 6, 10, 10]) == 0.75