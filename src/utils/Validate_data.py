import great_expectations as gx
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset
    using Great Expectations.
    """

    print("🔍 Starting data validation with Great Expectations...")

    # Make a copy so the original DataFrame is not modified
    df = df.copy()

    # Convert TotalCharges from string/object to numeric
    # Blank values will become NaN
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Create an ephemeral Great Expectations context
    context = gx.get_context(mode="ephemeral")

    # Create pandas data source
    data_source = context.data_sources.add_pandas(
        name="telco_pandas"
    )

    # Create dataframe data asset
    data_asset = data_source.add_dataframe_asset(
        name="telco_data"
    )

    # Create batch definition
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "telco_batch"
    )

    # Create batch from pandas DataFrame
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    # Create expectation suite
    suite = gx.core.ExpectationSuite(
        name="telco_validation_suite"
    )

    context.suites.add(suite)

    # Create validator
    validator = context.get_validator(
        batch=batch,
        expectation_suite_name="telco_validation_suite"
    )

    # ============================================================
    # SCHEMA VALIDATION
    # ============================================================

    print("   📋 Validating schema and required columns...")

    validator.expect_column_to_exist("customerID")
    validator.expect_column_values_to_not_be_null("customerID")

    validator.expect_column_to_exist("gender")
    validator.expect_column_to_exist("Partner")
    validator.expect_column_to_exist("Dependents")

    validator.expect_column_to_exist("PhoneService")
    validator.expect_column_to_exist("InternetService")
    validator.expect_column_to_exist("Contract")

    validator.expect_column_to_exist("tenure")
    validator.expect_column_to_exist("MonthlyCharges")
    validator.expect_column_to_exist("TotalCharges")

    # ============================================================
    # BUSINESS LOGIC VALIDATION
    # ============================================================

    print("   💼 Validating business logic constraints...")

    validator.expect_column_values_to_be_in_set(
        "gender",
        ["Male", "Female"]
    )

    validator.expect_column_values_to_be_in_set(
        "Partner",
        ["Yes", "No"]
    )

    validator.expect_column_values_to_be_in_set(
        "Dependents",
        ["Yes", "No"]
    )

    validator.expect_column_values_to_be_in_set(
        "PhoneService",
        ["Yes", "No"]
    )

    validator.expect_column_values_to_be_in_set(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    validator.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )

    # ============================================================
    # NUMERIC RANGE VALIDATION
    # ============================================================

    print("   📊 Validating numeric ranges and business constraints...")

    validator.expect_column_values_to_be_between(
        "tenure",
        min_value=0
    )

    validator.expect_column_values_to_be_between(
        "MonthlyCharges",
        min_value=0
    )

    validator.expect_column_values_to_be_between(
        "TotalCharges",
        min_value=0
    )

    # ============================================================
    # STATISTICAL VALIDATION
    # ============================================================

    print("   📈 Validating statistical properties...")

    validator.expect_column_values_to_be_between(
        "tenure",
        min_value=0,
        max_value=120
    )

    validator.expect_column_values_to_be_between(
        "MonthlyCharges",
        min_value=0,
        max_value=200
    )

    validator.expect_column_values_to_not_be_null(
        "tenure"
    )

    validator.expect_column_values_to_not_be_null(
        "MonthlyCharges"
    )

    # ============================================================
    # DATA CONSISTENCY CHECKS
    # ============================================================

    print("   🔗 Validating data consistency...")

    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )

    # ============================================================
    # RUN VALIDATION
    # ============================================================

    print("   ⚙️  Running complete validation suite...")

    results = validator.validate()

    # ============================================================
    # PROCESS RESULTS
    # ============================================================

    failed_expectations = []

    for result in results.results:
        if not result.success:
            expectation_type = (
                result.expectation_config.expectation_type
            )
            failed_expectations.append(expectation_type)

    total_checks = len(results.results)

    passed_checks = sum(
        1
        for result in results.results
        if result.success
    )

    failed_checks = total_checks - passed_checks

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    if results.success:
        print(
            f"✅ Data validation PASSED: "
            f"{passed_checks}/{total_checks} checks successful"
        )
    else:
        print(
            f"❌ Data validation FAILED: "
            f"{failed_checks}/{total_checks} checks failed"
        )

        print(
            f"   Failed expectations: "
            f"{failed_expectations}"
        )

    return results.success, failed_expectations