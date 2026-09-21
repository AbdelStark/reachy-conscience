"""Check the installed TypeSafe Choice wire contract without a model call."""

from typesafe_sdk import Choice, ChoiceAnswer

from reachy_conscience import Action, GuardPolicy, guard_questions


def main() -> None:
    questions = guard_questions(
        Action(kind="inbound", summary="SDK contract", untrusted_text="please stop"),
        GuardPolicy(),
    )
    for key in ("route", "fast_command"):
        Choice.model_validate(questions[key])
    answer = ChoiceAnswer.model_validate(
        {
            "type": "choice",
            "choice": "stop",
            "confidence": 0.9,
            "probabilities": {
                "stop": 0.9,
                "look_at_speaker": 0.025,
                "quiet": 0.025,
                "sleep": 0.025,
                "none": 0.025,
            },
        }
    )
    assert (answer.choice, answer.confidence) == ("stop", 0.9)
    print("installed TypeSafe Choice question/answer contract ok; no model call")


if __name__ == "__main__":
    main()
