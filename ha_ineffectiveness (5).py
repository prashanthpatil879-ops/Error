import re
import math
from pyspark.sql import DataFrame
from pyspark.sql import functions as F, Window
from typing import NamedTuple
from rda.engine.base_step import RDABaseStep
import rda.utils.spark_utils as su
import rda.utils.date_utils as du


class HAIneffectivenessInputs(NamedTuple):
    dv01_fi_df: DataFrame
    dv01_derivatives_df: DataFrame
    dv01_liab_post_hp_df: DataFrame
    dv01_mapping_nt_llp_df: DataFrame
    fx_rates_df: DataFrame
    hierarchy_mapping_df: DataFrame
    rf_ear_shocks_df: DataFrame
    avg_shocks_df: DataFrame
    ir_gamma_df: DataFrame


class HAIneffectiveness(RDABaseStep):

    # Columns carried through the unpivot, before the tenor buckets are stacked.
    ALM_KEY_COLS = ["ASSET_TYPE", "INCEPTION_DATE", "PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY", "SOURCE"]
    TENOR_COL_PATTERN = r"^\d+M$"

    # Everything is reported in CAD, so the IFS rates file carries no CAD row.
    BASE_CURRENCY = "CAD"


    # Columns on irr_ec_forecasting_ifs_fx_rates (curated). DATE is already a
    # proper date there, so no string parsing is needed.
    # BS_RATE = balance sheet (closing). IS_RATE = income statement (average).
    FX_CURRENCY_COL = "CODE"
    FX_RATE_COL = "BS_RATE"
    FX_DATE_COL = "DATE"

    # Tenors are doubles on both sides of the shocks join, so the key is rounded.
    TENOR_JOIN_DP = 6
    FALLBACK_SHOCK_CURRENCY = "USD"
    LOSSES_MULTIPLIER = 100

    # LOSSES_LIAB_WITH_GAMMA = GAMMA_MULTIPLIER * AVG_SHOCKS^2 * IR_GAMMA
    GAMMA_MULTIPLIER = 0.5

    # IR Gamma for an LLP + currency that has no row in the IR Gamma table.
    DEFAULT_IR_GAMMA = 0.0

    # Every measure column is written as 0 rather than null.
    NULL_FILL = 0.0

    ASSET_HOLDING_BOND = "Bond"
    ASSET_HOLDING_LIABILITY = "Liability"
    ASSET_HOLDING_MANUBANK = "Manubank"

    # Target column per asset holding on the losses table.
    LOSS_COLUMNS = {
        ASSET_HOLDING_LIABILITY: "LOSSES_LIAB_NO_GAMMA",
        ASSET_HOLDING_BOND: "LOSSES_HA_BOND",
        ASSET_HOLDING_MANUBANK: "LOSSES_MANUBANK",
    }

    LOSSES_GROUP_COLS = ["SECURITY_CURRENCY", "ASSET_HOLDING", "LOWEST_LEVEL_PORTFOLIO_NAME", "LEGAL_ID"]

    # Grain of the reporting losses table that feeds the percentile. The STM
    # takes "the 4750th value" of 5000, so this has to collapse to one row per
    # scenario - add SECURITY_CURRENCY here if per-currency percentiles are ever
    # wanted, but then the 4750th value no longer means the 95th percentile.
    REPORTING_LOSSES_GROUP_COLS = ["SECURITY_CURRENCY", "SCENARIO"]

    PERCENTILE_GROUP_COLS = ["SECURITY_CURRENCY"]

    # Loss measures carried on the losses and reporting losses tables, in order.
    LOSS_MEASURE_COLS = [
        "LOSSES_LIAB_NO_GAMMA",
        "LOSSES_LIAB_WITH_GAMMA",
        "LOSSES_HA_LIAB",
        "LOSSES_HA_BOND",
        "LOSSES_MANUBANK",
        "LOSSES_HA",
    ]

    # Percentile taken on the reporting losses. 0.95 x 5000 scenarios = the
    # 4750th value in ascending order, per the STM.
    PERCENTILE = 0.95

    # Source measure -> percentile column. Named CL95_* after the existing
    # CL95_EAR convention in non_ha.py, rather than a "95percentile_" prefix
    # which would put a digit at the front of the column name.
    PERCENTILE_COLUMNS = {
        "LOSSES_LIAB_NO_GAMMA": "95PERCENTILE_LIAB_NO_GAMMA",
        "LOSSES_LIAB_WITH_GAMMA": "95PERCENTILE_LIAB_WITH_GAMMA",
        "LOSSES_HA_LIAB": "95PERCENTILE_HA_LIAB",
        "LOSSES_HA_BOND": "95PERCENTILE_HA_BOND",
        "LOSSES_MANUBANK": "95PERCENTILE_MANUBANK",
        "LOSSES_HA": "95PERCENTILE_HA",
    }

    def read(self) -> HAIneffectivenessInputs:
        inputs = self._step_config.inputs
        last_quarter_end_date = self._date_ctx.asDict().get('PREVIOUSQUARTERENDDATE', None)
        self._logger.info(f"Reading inputs for HA Ineffectiveness process, Quarter End Date: {last_quarter_end_date}")

        if last_quarter_end_date is None:
            raise ValueError(f"Invalid date context entry for PREVIOUSQUARTERENDDATE in: {self._date_ctx.asDict()}")

        # ALM stamps a month's data on the 1st of the FOLLOWING month, so the
        # quarter ending 2026-03-31 arrives on the 2026-04-01 extract.
        next_month_first_day = du.get_next_month_first_day(last_quarter_end_date)

        # The curated FX table holds every month in one load, so the date filter
        # is what selects the quarter. read_watermark applies this filter BEFORE
        # taking MAX(INGESTION_TS), so the watermark is scoped to that month too.
        fx_filter = f"{self.FX_DATE_COL} = '{last_quarter_end_date}'"

        input_dfs = HAIneffectivenessInputs(
            dv01_fi_df = self._uc.read_latest(
                inputs.dv01_fi.table_name,
                filter=f"INCEPTION_DATE='{next_month_first_day}'",
                watermark_col=inputs.dv01_fi.watermark_col
            ),
            dv01_derivatives_df = self._uc.read_latest(
                inputs.dv01_derivatives.table_name,
                filter=f"INCEPTION_DATE='{next_month_first_day}'",
                watermark_col=inputs.dv01_derivatives.watermark_col
            ),
            dv01_liab_post_hp_df = self._uc.read_latest(
                inputs.dv01_liab_post_hp.table_name,
                filter=f"INCEPTION_DATE='{next_month_first_day}'",
                watermark_col=inputs.dv01_liab_post_hp.watermark_col
            ),
            dv01_mapping_nt_llp_df = self._uc.read_latest(
                inputs.dv01_mapping_nt_llp.table_name,
                watermark_col=inputs.dv01_mapping_nt_llp.watermark_col
            ),
            fx_rates_df = self._uc.read_latest(
                inputs.fx_rates.table_name,
                filter=fx_filter,
                watermark_col=inputs.fx_rates.watermark_col
            ),
            hierarchy_mapping_df = self._uc.read_latest(
                inputs.irr_mapping_heirarchy_mapping_latest.table_name,
                watermark_col=inputs.irr_mapping_heirarchy_mapping_latest.watermark_col
            ),
            rf_ear_shocks_df = self._uc.read_latest(
                inputs.rf_ear_shocks.table_name,
                filter=f"EFFECTIVE_DATE='{last_quarter_end_date}'",
                watermark_col=inputs.rf_ear_shocks.watermark_col
            ),
            avg_shocks_df = self._uc.read_latest(
                inputs.avg_shocks.table_name,
                filter=f"REPORTING_DATE='{last_quarter_end_date}'",
                watermark_col=inputs.avg_shocks.watermark_col
            ),
            ir_gamma_df = self._uc.read_latest(
                inputs.ir_gamma.table_name,
                filter=f"REPORTING_DATE='{last_quarter_end_date}'",
                watermark_col=inputs.ir_gamma.watermark_col
            )
        )

        return input_dfs

    def unpivot(self, df: DataFrame) -> DataFrame:
        tenors = [c for c in df.columns if re.fullmatch(self.TENOR_COL_PATTERN, c)]

        if not tenors:
            raise ValueError(f"No tenor columns matching {self.TENOR_COL_PATTERN} found in: {df.columns}")
        expr = "stack({0}, {1}) as (TENOR_CD, DV01_VALUE)".format(
            len(tenors), ", ".join([f"CAST({int(c[:-1]) / 12} AS DOUBLE), CAST(`{c}` AS DOUBLE)" for c in tenors])
        )
        return df.select(*self.ALM_KEY_COLS, F.expr(expr))


    def zero_fill(self, df: DataFrame, columns: list) -> DataFrame:
        """Write measures as 0 rather than null."""
        for column in columns:
            df = df.withColumn(column, F.coalesce(F.col(column), F.lit(self.NULL_FILL)))
        return df


    def validate_lookup(self, df: DataFrame, lookup_col: str, key_cols: list, source: str) -> None:
        unmatched = df.filter(F.col(lookup_col).isNull()).select(*key_cols).distinct().collect()

    def _ir_gamma_df(self, inputs: HAIneffectivenessInputs) -> DataFrame:
        """
        IR Gamma per LLP and currency from irr_ear_reporting_ha_ir_gamma.
        GTD_TAM_SEGMENT on that table carries the LLP name, so it joins straight
        to LOWEST_LEVEL_PORTFOLIO_NAME with no hierarchy lookup in between.
        """
        return (
            inputs.ir_gamma_df
            .select(
                F.trim(F.col("GTD_TAM_SEGMENT")).alias("GAMMA_LLP"),
                F.upper(F.trim(F.col("CURRENCY_CODE"))).alias("GAMMA_CURRENCY"),
                F.col("IR_GAMMA").cast("double").alias("IR_GAMMA")
            )
            .dropDuplicates(["GAMMA_LLP", "GAMMA_CURRENCY"])
        )


    def _avg_shocks_df(self, inputs: HAIneffectivenessInputs) -> DataFrame:
        """One average shock per currency and scenario. SCENARIO is space padded
        in the source, so it is trimmed before casting."""
        return (
            inputs.avg_shocks_df
            .select(
                F.upper(F.trim(F.col("CURRENCY"))).alias("AVG_CURRENCY"),
                F.trim(F.col("SCENARIO").cast("string")).cast("int").alias("AVG_SCENARIO"),
                F.col("AVG_SHOCKS").cast("double").alias("AVG_SHOCKS")
            )
            .dropDuplicates(["AVG_CURRENCY", "AVG_SCENARIO"])
        )


    def _llp_mapping_df(self, inputs: HAIneffectivenessInputs) -> DataFrame:
        return (
            inputs.dv01_mapping_nt_llp_df
            .select(
                F.upper(F.trim(F.col("PROGRAM_CODE"))).alias("PROGRAM_CODE"),
                F.col("LEGAL_ID").cast("int").alias("LEGAL_ID"),
                F.upper(F.trim(F.col("SECURITY_CURRENCY"))).alias("SECURITY_CURRENCY"),
                F.col("LOWEST_LEVEL_PORTFOLIO_NAME"),
            )
            .dropDuplicates(["PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY"])
        )

    def calculate_partial_dv01(self, inputs: HAIneffectivenessInputs) -> DataFrame:
        self._logger.info("Starting HA Ineffectiveness - Partial DV01 calculation process...")
        last_quarter_end_date = self._date_ctx.asDict().get('PREVIOUSQUARTERENDDATE')

        dv01_unpivot_df = (
            self.unpivot(inputs.dv01_fi_df)
            .unionByName(self.unpivot(inputs.dv01_derivatives_df))
            .unionByName(self.unpivot(inputs.dv01_liab_post_hp_df))
            .withColumn("DV01_VALUE", F.coalesce(F.col("DV01_VALUE"), F.lit(0.0)))
            .withColumn("INCEPTION_DATE", F.to_date(F.col("INCEPTION_DATE")))
            .withColumn("PROGRAM_CODE", F.upper(F.trim(F.col("PROGRAM_CODE"))))
            .withColumn("SECURITY_CURRENCY", F.upper(F.trim(F.col("SECURITY_CURRENCY"))))
            .withColumn("LEGAL_ID", F.col("LEGAL_ID").cast("int"))
        )

        # TOTAL_DV01 = sum of DV01 across the three ALM datasets, per tenor. The
        # datasets are disjoint on PROGRAM_CODE (GH_01 only in liab_post_hp +
        # derivatives, the rest only in fi + derivatives), so this reproduces the
        # per-combination source rules in HA DV01 mapping_NT_LLP.xlsx.
        # The reporting date is supplied by the REPORTING_DATE audit column that
        # RDABaseStep.write adds, so it is not carried as a business column.
        total_dv01_df = (
            dv01_unpivot_df
            .groupBy("PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY", "TENOR_CD")
            .agg( F.sum("DV01_VALUE").alias("TOTAL_DV01") )
        )

        total_dv01_df = self.cache_df(total_dv01_df, "total_dv01_df")

        # LLP from the mapping, then TAX_RATE from the hierarchy mapping keyed on
        # that LLP. su.normalize_text absorbs the case difference between the
        # mapping ("ManuBank") and the hierarchy table ("Manubank").
        tax_rate_df = (
            inputs.hierarchy_mapping_df
            .withColumn("rnk", F.dense_rank().over(Window.partitionBy(su.normalize_text(F.col("LOWEST_LEVEL_PORTFOLIO_NAME"))).orderBy(F.col("REPORTING_DATE_KEY").desc())))
            .filter( F.col("rnk") == 1 )
            .select(
                su.normalize_text(F.col("LOWEST_LEVEL_PORTFOLIO_NAME")).alias("LLP_KEY"),
                F.col("TAX_RATE").cast("double").alias("TAX_RATE")
            )
            .dropDuplicates(["LLP_KEY"])
        )

        dv01_with_llp_df = (
            total_dv01_df.alias("d")
            .join(
                self._llp_mapping_df(inputs).alias("m"),
                on = ["PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY"],
                how = "left"
            )
            .select("d.*", "m.LOWEST_LEVEL_PORTFOLIO_NAME")
        )
        dv01_with_tax_df = (
            dv01_with_llp_df.alias("d")
            .join(
                tax_rate_df.alias("t"),
                on = su.normalize_text(F.col("d.LOWEST_LEVEL_PORTFOLIO_NAME")) == F.col("t.LLP_KEY"),
                how = "left"
            )
            .select("d.*", "t.TAX_RATE")
        )
        fx_rate_df = (
            inputs.fx_rates_df
            .select(
                F.upper(F.trim(F.col(self.FX_CURRENCY_COL))).alias("FX_CURRENCY"),
                F.col(self.FX_RATE_COL).cast("double").alias("FX_RATE")
            )
            .dropDuplicates(["FX_CURRENCY"])
        )

        dv01_with_fx_df = (
            dv01_with_tax_df.alias("d")
            .join(
                fx_rate_df.alias("f"),
                on = F.col("d.SECURITY_CURRENCY") == F.col("f.FX_CURRENCY"),
                how = "left"
            )
            .select("d.*", "f.FX_RATE")
            # The reporting currency has no row in the rates file; give it 1.
            # Scoped to BASE_CURRENCY only - any other unmatched currency stays
            # null and is reported below, because defaulting something like JPY
            # to 1.0 would misstate it by roughly 100x.
            .withColumn(
                "FX_RATE",
                F.when( F.col("FX_RATE").isNull() & (F.col("SECURITY_CURRENCY") == F.lit(self.BASE_CURRENCY)), F.lit(1.0) )
                 .otherwise( F.col("FX_RATE") )
            )
        )

        partial_dv01_df = (
            dv01_with_fx_df
            .withColumn(
                "ASSET_HOLDING",
                F.when( F.col("PROGRAM_CODE").isin("GH_14", "GH_09", "GH_02"), self.ASSET_HOLDING_BOND )
                 .when( F.col("PROGRAM_CODE") == "GH_01", self.ASSET_HOLDING_LIABILITY )
                 .when( F.col("PROGRAM_CODE") == "MB_01", self.ASSET_HOLDING_MANUBANK )
            )
            .withColumn("SUM_PARTIAL_DV01", F.col("TOTAL_DV01") * F.col("FX_RATE") * (1 - F.col("TAX_RATE")))
            .select(
                "SECURITY_CURRENCY", "ASSET_HOLDING", "TENOR_CD",
                "PROGRAM_CODE", "LOWEST_LEVEL_PORTFOLIO_NAME", "LEGAL_ID",
                "TOTAL_DV01", "FX_RATE", "TAX_RATE", "SUM_PARTIAL_DV01"
            )
        )
        partial_dv01_df = self.zero_fill(partial_dv01_df, ["TOTAL_DV01", "FX_RATE", "TAX_RATE", "SUM_PARTIAL_DV01"])

        # Left joins never drop rows, so the grain must still equal TOTAL_DV01.
        # A mismatch means a mapping has duplicate keys and DV01 is double counted.
        if partial_dv01_df.count() != total_dv01_df.count():
            raise ValueError("Row count changed across the lookups - a mapping has duplicate keys")

        partial_dv01_df = self.cache_df(partial_dv01_df, "partial_dv01_df")

        #self.write(partial_dv01_df, self._step_config.step_partial_dv01.table_name, self._step_config.step_partial_dv01.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        self._logger.info("HA Ineffectiveness - Partial DV01 calculation completed")
        return partial_dv01_df
    
    def calculate_losses(self, inputs: HAIneffectivenessInputs, partial_dv01_df: DataFrame) -> DataFrame:
        self._logger.info("Starting HA Ineffectiveness - Losses calculation process...")
 
        # Collapse to one DV01 per key + tenor before the shocks are applied, in
        # case a key ever carries more than one PROGRAM_CODE.
        dv01_df = (
            partial_dv01_df
            .groupBy(*self.LOSSES_GROUP_COLS, "TENOR_CD")
            .agg( F.sum("SUM_PARTIAL_DV01").alias("SUM_PARTIAL_DV01") )
            .withColumn("TENOR_KEY", F.round(F.col("TENOR_CD"), self.TENOR_JOIN_DP))
        )
 
        rf_ear_shocks_base_df = (
            inputs.rf_ear_shocks_df
            .select(
                F.upper(F.trim(F.col("CURRENCY"))).alias("SHOCK_CURRENCY"),
                F.round(F.col("TENOR").cast("double"), self.TENOR_JOIN_DP).alias("TENOR_KEY"),
                F.col("SCENARIO").cast("int").alias("SCENARIO"),
                F.col("SHOCKS").cast("double").alias("SHOCKS")
            )
            .dropDuplicates(["SHOCK_CURRENCY", "TENOR_KEY", "SCENARIO"])
        )
 
        shock_currencies = sorted(r['SHOCK_CURRENCY'] for r in rf_ear_shocks_base_df.select("SHOCK_CURRENCY").distinct().collect() )
        if self.FALLBACK_SHOCK_CURRENCY not in shock_currencies:
            self._logger.warning(f"The Fallback shock currency {self.FALLBACK_SHOCK_CURRENCY} is not in the shock table")
 
        dv01_df = dv01_df.withColumn("SHOCK_CURRENCY_KEY", F.when(F.col("SECURITY_CURRENCY").isin(shock_currencies), F.col("SECURITY_CURRENCY"))
                         .otherwise(F.lit(self.FALLBACK_SHOCK_CURRENCY)))
 
        borrowed = sorted(r["SECURITY_CURRENCY"] for r in dv01_df.filter(F.col("SECURITY_CURRENCY") != F.col("SHOCK_CURRENCY_KEY")).select("SECURITY_CURRENCY").distinct().collect())
        if borrowed:
            self._logger.warning(f"No shock for {borrowed} - using {self.FALLBACK_SHOCK_CURRENCY} shocks for them." f"Their DV01 is still converted at their own FX rate.")
 
        # Each tenor is multiplied by the shock for THAT SAME tenor and scenario;
        # the products are then summed across the curve and scaled by 100.
        # dv01_df is tiny against millions of shock rows, so it is broadcast.
        losses_base_df = (
            F.broadcast(dv01_df).alias("d")
            .join(
                rf_ear_shocks_base_df.alias("s"),
                on = [
                    F.col("d.SHOCK_CURRENCY_KEY") == F.col("s.SHOCK_CURRENCY"),
                    F.col("d.TENOR_KEY") == F.col("s.TENOR_KEY")
                ],
                how = "left"
            )
            .select("d.*", "s.SCENARIO", "s.SHOCKS")
        )
 
        unmatched = (
            losses_base_df.filter( F.col("SHOCKS").isNull() )
            .groupBy("SECURITY_CURRENCY")
            .agg( F.sort_array(F.collect_set("TENOR_CD")).alias("TENORS") )
            .collect()
        )
 
        # All three loss columns come from one pass: sum only the products
        # belonging to that asset holding. A key with no rows for an asset
        # holding gets null, not zero.
        losses_amt_df = (
            losses_base_df
            .filter( F.col("SCENARIO").isNotNull() )
            .withColumn("TENOR_PRODUCT", F.col("SUM_PARTIAL_DV01") * F.col("SHOCKS"))
            .groupBy(*self.LOSSES_GROUP_COLS, "SCENARIO")
            .agg(
                *[
                    ( F.sum(F.when(F.col("ASSET_HOLDING") == holding, F.col("TENOR_PRODUCT"))) * F.lit(self.LOSSES_MULTIPLIER) ).alias(column)
                    for holding, column in self.LOSS_COLUMNS.items()
                ],
                *[
                    F.count(F.when(F.col("ASSET_HOLDING") == holding, F.col("TENOR_PRODUCT"))).alias(f"{column}_TENORS")
                    for holding, column in self.LOSS_COLUMNS.items()
                ]
            )
        )
 
        # Every populated column must be built from the full tenor curve. A count
        # between 1 and expected-1 means a tenor was lost in the shocks join.
        expected_tenors = partial_dv01_df.select("TENOR_CD").distinct().count()
 
        incomplete_curve = None
        for column in self.LOSS_COLUMNS.values():
            check = ~F.col(f"{column}_TENORS").isin(0, expected_tenors)
            incomplete_curve = check if incomplete_curve is None else (incomplete_curve | check)
 
        partial_curves = losses_amt_df.filter(incomplete_curve).count()
 
        # --- IR Gamma per LLP ------------------------------------------------
        # Joined on LLP *and* currency: the same LLP appears under several
        # currencies (MLJ JPY-Guaranteed maps from AUD, CAD, EUR, GBP, NOK and
        # USD), so joining on LLP alone would multiply every row.
        ir_gamma_df = self._ir_gamma_df(inputs)

        losses_gamma_df = (
            losses_amt_df.alias("l")
            .join(
                ir_gamma_df.alias("g"),
                on = [
                    F.col("l.SECURITY_CURRENCY") == F.col("g.GAMMA_CURRENCY"),
                    su.normalize_text(F.col("l.LOWEST_LEVEL_PORTFOLIO_NAME")) == su.normalize_text(F.col("g.GAMMA_LLP"))
                ],
                how = "left"
            )
            .select("l.*", "g.IR_GAMMA")
        )

        # An LLP + currency with no row in the IR Gamma table gets
        # DEFAULT_IR_GAMMA rather than a null, so its with-gamma loss is zero.
        no_gamma = (
            losses_gamma_df.filter( F.col("IR_GAMMA").isNull() )
            .select("SECURITY_CURRENCY", "LOWEST_LEVEL_PORTFOLIO_NAME").distinct().collect()
        )

        losses_gamma_df = losses_gamma_df.withColumn(
            "IR_GAMMA", F.coalesce(F.col("IR_GAMMA"), F.lit(self.DEFAULT_IR_GAMMA))
        )

        # --- Average shock per currency and scenario -------------------------
        avg_shocks_df = self._avg_shocks_df(inputs)

        avg_currencies = sorted(
            r["AVG_CURRENCY"] for r in avg_shocks_df.select("AVG_CURRENCY").distinct().collect()
        )
        losses_gamma_df = losses_gamma_df.withColumn(
            "AVG_CURRENCY_KEY",
            F.when( F.col("SECURITY_CURRENCY").isin(avg_currencies), F.col("SECURITY_CURRENCY") )
             .otherwise( F.lit(self.FALLBACK_SHOCK_CURRENCY) )
        )

        borrowed_avg = sorted(
            r["SECURITY_CURRENCY"] for r in
            losses_gamma_df.filter( F.col("SECURITY_CURRENCY") != F.col("AVG_CURRENCY_KEY") )
                           .select("SECURITY_CURRENCY").distinct().collect()
        )
        losses_avg_df = (
            losses_gamma_df.alias("l")
            .join(
                avg_shocks_df.alias("a"),
                on = [
                    F.col("l.AVG_CURRENCY_KEY") == F.col("a.AVG_CURRENCY"),
                    F.col("l.SCENARIO") == F.col("a.AVG_SCENARIO")
                ],
                how = "left"
            )
            .select("l.*", "a.AVG_SHOCKS")
        )

        if losses_avg_df.count() != losses_amt_df.count():
            raise ValueError("Row count changed across the IR Gamma / average shock joins - a mapping has duplicate keys")

        # --- The gamma measures ----------------------------------------------
        losses_with_llp_df = (
            losses_avg_df
            # 0.5 * (average shock for that scenario)^2 * IR Gamma for that LLP.
            # Scoped to Liability rows so it lines up with LOSSES_LIAB_NO_GAMMA,
            # which is itself only populated for Liability.
            .withColumn(
                "LOSSES_LIAB_WITH_GAMMA",
                F.when(
                    F.col("ASSET_HOLDING") == self.ASSET_HOLDING_LIABILITY,
                    F.lit(self.GAMMA_MULTIPLIER) * F.pow(F.col("AVG_SHOCKS"), F.lit(2)) * F.col("IR_GAMMA")
                )
            )
            .withColumn("LOSSES_HA_LIAB", F.col("LOSSES_LIAB_NO_GAMMA") + F.col("LOSSES_LIAB_WITH_GAMMA"))
            # Same null-as-zero rule as the reporting losses step. Each row here
            # carries one asset holding, so LOSSES_HA is that row's contribution
            # to the currency total rather than the total itself.
            .withColumn(
                "LOSSES_HA",
                F.coalesce(F.col("LOSSES_HA_LIAB"), F.lit(0.0))
                + F.coalesce(F.col("LOSSES_HA_BOND"), F.lit(0.0))
                + F.coalesce(F.col("LOSSES_MANUBANK"), F.lit(0.0))
            )
            .select(
                "SECURITY_CURRENCY",
                "ASSET_HOLDING",
                "LOWEST_LEVEL_PORTFOLIO_NAME",
                "LEGAL_ID",
                "SCENARIO",
                "IR_GAMMA",
                "AVG_SHOCKS",
                *self.LOSS_MEASURE_COLS
            )
        )

        losses_with_llp_df = self.zero_fill(losses_with_llp_df, ["IR_GAMMA", "AVG_SHOCKS", *self.LOSS_MEASURE_COLS])
        losses_with_llp_df = self.cache_df(losses_with_llp_df, "losses_with_llp_df")
 
        #self.write(losses_with_llp_df, self._step_config.step_losses_with_llp.table_name, self._step_config.step_losses_with_llp.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        self._logger.info("HA Ineffectiveness - Losses calculation completed")
        return losses_with_llp_df
 
    def calculate_reporting_losses(self, losses_with_llp_df: DataFrame) -> DataFrame:
        """Collapse the per-LLP losses to one row per scenario for the percentile."""
        self._logger.info("Starting HA Ineffectiveness - Reporting losses calculation process...")
 
        reporting_losses_df = (
            losses_with_llp_df
            .groupBy(*self.REPORTING_LOSSES_GROUP_COLS)
            .agg(
                F.sum("LOSSES_LIAB_NO_GAMMA").alias("LOSSES_LIAB_NO_GAMMA"),
                F.sum("LOSSES_LIAB_WITH_GAMMA").alias("LOSSES_LIAB_WITH_GAMMA"),
                F.sum("LOSSES_HA_BOND").alias("LOSSES_HA_BOND"),
                F.sum("LOSSES_MANUBANK").alias("LOSSES_MANUBANK")
            )
            # Re-derive rather than summing the row level values, so the totals
            # stay consistent with their components after aggregation.
            .withColumn("LOSSES_HA_LIAB", F.col("LOSSES_LIAB_NO_GAMMA") + F.col("LOSSES_LIAB_WITH_GAMMA"))
            # LOSSES_HA totals the three legs, treating a null leg as zero so a
            # currency with no position in one leg still gets a total.
            .withColumn(
                "LOSSES_HA",
                F.coalesce(F.col("LOSSES_HA_LIAB"), F.lit(0.0))
                + F.coalesce(F.col("LOSSES_HA_BOND"), F.lit(0.0))
                + F.coalesce(F.col("LOSSES_MANUBANK"), F.lit(0.0))
            )
            .select(*self.REPORTING_LOSSES_GROUP_COLS, *self.LOSS_MEASURE_COLS)
        )
 
        reporting_losses_df = self.zero_fill(reporting_losses_df, self.LOSS_MEASURE_COLS)
        reporting_losses_df = self.cache_df(reporting_losses_df, "reporting_losses_df")
 
        #self.write(reporting_losses_df, self._step_config.step_reporting_losses.table_name, self._step_config.step_reporting_losses.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        self._logger.info("HA Ineffectiveness - Reporting losses calculation completed")
        return reporting_losses_df
 
    def calculate_percentile(self, reporting_losses_df: DataFrame) -> DataFrame:
        """95th percentile per currency: sort ascending, take the 4750th of 5000."""
        self._logger.info("Starting HA Ineffectiveness - Percentile calculation process...")
 
        # Collect (value, scenario) pairs per measure so the scenario that
        # produced the percentile can be reported alongside it. sort_array on a
        # struct orders by the first field, so this sorts by value ascending and
        # breaks ties on scenario. The when() yields null for a measure with no
        # value and collect_list drops nulls, so a measure still on hold gives an
        # empty array rather than nulls sorted to the front.
        sorted_df = (
            reporting_losses_df
            .groupBy(*self.PERCENTILE_GROUP_COLS)
            .agg(
                F.countDistinct("SCENARIO").alias("SCENARIO_COUNT"),
                *[
                    F.sort_array(F.collect_list(F.when(F.col(measure).isNotNull(), F.col(measure)))).alias(f"{measure}_SORTED")
                    for measure in self.LOSS_MEASURE_COLS
                ]
            )
            # Rank is derived per currency, so a currency with a short scenario
            # set is ranked on its own count rather than a hardcoded 4750.
            .withColumn("PERCENTILE_RANK", F.ceil(F.col("SCENARIO_COUNT") * F.lit(self.PERCENTILE)).cast("int"))
        )
 
        percentile_df = sorted_df.select(
            *self.PERCENTILE_GROUP_COLS,
            *[
                F.when(
                    F.size(F.col(f"{measure}_SORTED")) >= F.col("PERCENTILE_RANK"),
                    F.element_at(F.col(f"{measure}_SORTED"), F.col("PERCENTILE_RANK"))
                ).alias(column)
                for measure, column in self.PERCENTILE_COLUMNS.items()
            ]
        ).orderBy(*self.PERCENTILE_GROUP_COLS)

        percentile_df = self.zero_fill(percentile_df, list(self.PERCENTILE_COLUMNS.values()))

        percentile_df = self.cache_df(percentile_df, "percentile_df")
 
        #self.write(percentile_df, self._step_config.step_percentile.table_name, self._step_config.step_percentile.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        self._logger.info("HA Ineffectiveness - Percentile calculation completed")
        return percentile_df