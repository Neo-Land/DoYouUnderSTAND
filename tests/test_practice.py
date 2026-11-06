from dyu.practice import CLOSE, CORRECT, WRONG, PracticeSession, easiest_first


def guesses(*words):
    return [(w, 0.5 / (i + 1)) for i, w in enumerate(words)]


def test_grades_correct_close_and_wrong():
    s = PracticeSession(["WATER", "HELLO"])
    g = s.grade(guesses("WATER", "MOTHER"))
    assert g.outcome == CORRECT and g.rank == 1
    g = s.grade(guesses("MOTHER", "EAT", "WATER"))
    assert g.outcome == CLOSE and g.rank == 3 and g.guess == "MOTHER"
    g = s.grade(guesses("MOTHER", "EAT"))
    assert g.outcome == WRONG and g.rank is None
    assert (s.attempts, s.correct, s.close) == (3, 1, 1)


def test_streak_resets_on_a_miss_and_best_is_kept():
    s = PracticeSession(["WATER"])
    for _ in range(3):
        s.grade(guesses("WATER"))
    s.grade(guesses("MOTHER"))
    s.grade(guesses("WATER"))
    assert s.streak == 1 and s.best_streak == 3
    assert abs(s.accuracy - 4 / 5) < 1e-9


def test_next_wraps_around():
    s = PracticeSession(["A", "B"])
    assert s.target == "A" and s.position == 1
    assert s.next() == "B" and s.position == 2
    assert s.next() == "A" and s.position == 1


def test_easiest_first_order_and_display_names(tmp_path):
    csv = tmp_path / "per_sign.csv"
    csv.write_text("gloss,accuracy,videos\nHARD,0.4,10\nEASY1,1.0,10\nMID,0.7,10\n", encoding="utf-8")
    order = easiest_first(csv, ["HARD", "MID", "EASY1", "UNSEEN"], lambda g: g.rstrip("0123456789"))
    assert order == ["EASY", "MID", "HARD", "UNSEEN"]          # unknown accuracy goes last


def test_easiest_first_without_a_report_keeps_the_vocabulary_order(tmp_path):
    assert easiest_first(tmp_path / "missing.csv", ["B", "A"], str) == ["B", "A"]
