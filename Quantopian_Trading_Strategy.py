from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data import Fundamentals

from quantopian.pipeline.factors import SimpleMovingAverage, RollingLinearRegressionOfReturns, Returns
from quantopian.pipeline.filters.morningstar import Q1500US, Q500US

from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.classifiers.morningstar import Sector


# from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data.morningstar import operation_ratios
from quantopian.pipeline.data import morningstar

# EventVestor Earnings Calendar free from 01 Feb 2007 to Dec 2017.
from quantopian.pipeline.factors.eventvestor import (
    BusinessDaysUntilNextEarnings,
    BusinessDaysSincePreviousEarnings,
)

# EventVestor Mergers & Acquisitions free from 01 Feb 2007 to 2017.
from quantopian.pipeline.filters.eventvestor import IsAnnouncedAcqTarget

from quantopian.pipeline.factors import BusinessDaysSincePreviousEvent

import quantopian.optimize as opt

import numpy as np
import pandas as pd

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
# TOTAL_POSITIONS = 1000
MAX_SHORT_POSITION_SIZE = 0.05  # 0.01 (1% or 5%)
MAX_LONG_POSITION_SIZE = 0.05  # 0.01
# MAX_SHORT_POSITION_SIZE = 2 / TOTAL_POSITIONS  #Dividing Equally
# MAX_LONG_POSITION_SIZE = 2 / TOTAL_POSITIONS  

# Risk Exposures
MAX_SECTOR_EXPOSURE = 0.01
MAX_BETA_EXPOSURE = 0.20


 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    # Rebalance every day, 10 mins after market open.
    schedule_function(my_rebalance, date_rules.week_start(), time_rules.market_open(minutes=10),half_days=False)
    
    # schedule_function(record_logs, date_rules.every_day(), time_rules.market_close())

     
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(), 'my_pipeline')
    
    # set_slippage(slippage.FixedSlippage(spread=0.01))
    set_commission(commission.PerShare(cost=0.0001))
    

class Momentum(CustomFactor):
    """ Conventional Momentum factor """
    inputs = [USEquityPricing.close,
                  Returns(window_length=126)]
    window_length = 252
# /np.nanstd(returns, axis=0)
    def compute(self, today, assets, out, prices, returns):
        out[:] = ((prices[-21] - prices[-252])/prices[-252] - (prices[-1] - prices[-21])/prices[-21])

class Liquidity(CustomFactor):   
    inputs = [USEquityPricing.volume,morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    def compute(self, today, assets, out, volume, shares):       
        out[:] = volume[-1]/shares[-1]

class Downside_Volatility(CustomFactor):  
    # Use returns as the input  
    inputs = [Returns(window_length=2)]  
    window_length = 10  
    def compute(self, today, assets, out, returns):  
        # set any returns greater than 0 to NaN so they are excluded from our calculations  
        returns[returns > 0] = np.nan  
        out[:] = np.nanstd(returns, axis=0)

def make_pipeline():
    """
     Defining out trading universe and alpha factor for long and short equities 
    """
    
    operation_margin = operation_ratios.operation_margin.latest
    revenue_growth = operation_ratios.revenue_growth.latest
    value = (Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest)
    roe = Fundamentals.roe.latest
    momentum = Momentum()
    liquidity = Liquidity()
    free_cash_flows = morningstar.valuation_ratios.fcf_yield.latest
    earnings_yield = morningstar.valuation_ratios.earning_yield.latest
    
    

    # Filter for stocks that are announced acquisition target.
    not_announced_acq_target = ~IsAnnouncedAcqTarget()
    
    mkt_cap_filter = morningstar.valuation.market_cap.latest >= 500000000
    # dividend_filter = Fundamentals.valuation_ratios.dividend_yield >= 0.02
    
    # Our universe is made up of stocks that have a non-null factors & with market cap over 500 Million $, are not announced 
    # acquisition targets, and are in the Q1500US.
    universe = (
        QTradableStocksUS() 
        # & not_near_earnings_announcement
        & not_announced_acq_target
        & earnings_yield.notnull()
        & free_cash_flows.notnull()
        & value.notnull()
        & roe.notnull()
        & mkt_cap_filter
        & momentum.notnull()
    )
    
    # combined_factor = roe*0
    # for factor in factors:
    #     combined_factor += factor.rank(mask=universe)
        
    
    # prediction_rank_quantiles = prediction_quality.quantiles(5)
    
    # longs = prediction_rank_quantiles.eq(4)
    # shorts = prediction_rank_quantiles.eq(0)
    
    combined_factor = (
        earnings_yield.winsorize(min_percentile=0.05, max_percentile=0.95).zscore(mask=universe) + 
        free_cash_flows.winsorize(min_percentile=0.05, max_percentile=0.95).zscore(mask=universe) + 
        value.winsorize(min_percentile=0.05, max_percentile=0.95).zscore(mask=universe) + roe.winsorize(min_percentile=0.05, max_percentile=0.95).zscore(mask=universe) + momentum.winsorize(min_percentile=0.05, max_percentile=0.95).zscore(mask=universe)                                                              
    )
    combined_factor_quantiles = combined_factor.quantiles(10)
    
    longs = combined_factor_quantiles.eq(9)
    shorts = combined_factor_quantiles.eq(0)
    # longs = combined_factor.top(2*TOTAL_POSITIONS//3, mask=universe)
    # shorts = combined_factor.bottom(TOTAL_POSITIONS//3, mask=universe)
    
    # We will take market beta into consideration when placing orders in our algorithm.
    beta = RollingLinearRegressionOfReturns(
                    target=sid(8554),
                    returns_length=5,
                    regression_length=260,
                    mask=(longs | shorts)
    ).beta
    
    # I calculated the market beta using rolling window regression using Bloomberg's computation.
    # Ref: https://guides.lib.byu.edu/c.php?g=216390&p=1428678
    bb_beta = (0.66 * beta) + (0.33 * 1.0)
 
    ## create pipeline
    columns = {
        'longs': longs,
        'shorts': shorts,
        'market_beta': bb_beta,
        'sector': Sector(),
        'combined_factor' : combined_factor,
    }
    pipe = Pipeline(columns=columns, screen=(longs | shorts))
 
    return pipe
    
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.pipeline_output = pipeline_output('my_pipeline')
  
    # These are the securities that we are interested in trading each day.
    context.securities = context.pipeline_output.index.tolist()
    
    # Replace NaN beta values with 1.0.
    context.market_beta = context.pipeline_output.market_beta.fillna(1.0)
    
 
def my_rebalance(context,data):
    """
    Place orders according to our schedule_function() timing.
    """
    
    # Compute our portfolio weights.
    long_secs = context.pipeline_output[context.pipeline_output['longs']].index
    long_weight = 0.5 / len(long_secs) 
    
    short_secs = context.pipeline_output[context.pipeline_output['shorts']].index
    short_weight = -0.5 / len(short_secs)
    
    combined_weights = {}
    
    # Open our long positions.
    for security in long_secs:
        combined_weights[security] = long_weight
    
    # Open our short positions.
    for security in short_secs:
        combined_weights[security] = short_weight
    
    # Sets our objective to maximize alpha based on the weights we receive from our factor.
    objective = opt.MaximizeAlpha(context.pipeline_output['combined_factor'])
    # objective = opt.TargetWeights(combined_weights)

    # Constraints
    # -----------
    # Constrain our gross leverage to 1.0 or less. This means that the absolute
    # value of our long and short positions should not exceed the value of our
    # portfolio.
    constrain_gross_leverage = opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)
    
    # Constrain individual position size to no more than a fixed percentage 
    # of our portfolio. Because our alphas are so widely distributed, we 
    # should expect to end up hitting this max for every stock in our universe.
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -MAX_SHORT_POSITION_SIZE,
        MAX_LONG_POSITION_SIZE,
    )

    # Constrain ourselves to allocate the same amount of capital to 
    # long and short positions.
    dollar_neutral = opt.DollarNeutral(tolerance=0.05)
    
    #Market neutrality constraint. Our portfolio should not be over-exposed
    #to the performance of the market (beta-to-spy).
    market_neutral = opt.FactorExposure(
        loadings=pd.DataFrame({'market_beta': context.market_beta}),
        min_exposures={'market_beta': -MAX_BETA_EXPOSURE},
        max_exposures={'market_beta': MAX_BETA_EXPOSURE},
    )
    
    # Sector neutrality constraint. Our portfolio should not be over-
    # exposed to any particular sector.
    sector_neutral = opt.NetGroupExposure.with_equal_bounds(
            labels=context.pipeline_output.sector,
            min=-MAX_SECTOR_EXPOSURE,
            max=MAX_SECTOR_EXPOSURE,
    )
    
    order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            dollar_neutral,
            market_neutral,
            sector_neutral,
        ],
        universe=context.securities,
    )
    