from modelrisk import ValidationEngine

engine = ValidationEngine()

report = engine.performance.validate(
    data=df,
    target="default_flag",
    prediction="pd"
)
