"""
Agent Optimization Testing Framework

This module provides a "participant as judge" testing framework for AI agents.
Instead of automated grading, it displays reference answers alongside agent outputs,
allowing participants to self-assess and iterate on their agent designs.

Typical Usage:
    from agent_optimization import run_optimization_tests
    
    # Run all test scenarios for your agent
    run_optimization_tests(
        agent=my_agent,
        agent_type="market_intelligence",
        show_hints=False
    )

Author: Talent Arena 2026
License: MIT
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import time
import sys
from io import StringIO


# =============================================================================
# SCENARIO DATA STRUCTURES
# =============================================================================

@dataclass
class OptimizationScenario:
    """
    Represents a single optimization test scenario.
    
    Attributes:
        name: Short, descriptive name of the scenario (e.g., "Basic Stock Analysis")
        description: What aspect of the agent this scenario tests
        query: The input query to pass to the agent
        expected_behavior: List of expected behaviors (tool usage, workflow steps, etc.)
        reference_answer: Example of a high-quality agent response
        checklist: List of evaluation criteria for participant self-assessment
        hints: Optional list of optimization hints to help participants improve
    
    Example:
        scenario = OptimizationScenario(
            name="Basic Workflow Test",
            description="Tests core functionality and output formatting",
            query="AAPL",
            expected_behavior=[
                "Call get_stock_price once",
                "Provide formatted output"
            ],
            reference_answer="📊 AAPL: $182.45 (+0.48%)",
            checklist=[
                "✅ Includes price data",
                "✅ Professional formatting"
            ],
            hints=["Add output template to system prompt"]
        )
    """
    name: str
    description: str
    query: str
    expected_behavior: List[str]
    reference_answer: str
    checklist: List[str]
    hints: Optional[List[str]] = None


# =============================================================================
# MARKET INTELLIGENCE AGENT SCENARIOS
# =============================================================================

_MARKET_SCENARIO_1 = OptimizationScenario(
    name="Basic Stock Analysis (Small Change)",
    description="Tests basic workflow, efficient tool usage, and output formatting",
    query="AAPL",
    expected_behavior=[
        "Call get_stock_price('AAPL') exactly once",
        "Analyze the price change percentage",
        "Skip news search if absolute change < 2%",
        "Provide formatted final answer with price, change, and brief analysis",
        "Complete in 2-3 ReAct iterations"
    ],
    reference_answer="""📊 AAPL Stock Analysis

Current Price: $182.45
Change: +$0.87 (+0.48%)

Analysis: Apple stock is up slightly today with a modest gain of 0.48%. 
This represents normal market movement and doesn't indicate any significant 
development. The stock continues trading within its recent range.""",
    checklist=[
        "✅ Includes current price with $ sign and proper formatting",
        "✅ Shows both change amount and percentage",
        "✅ Provides brief analysis/context (not just data)",
        "✅ Did NOT search news (change < 2%)",
        "✅ Completed efficiently (2-3 iterations)",
        "✅ Professional, clear format with sections"
    ],
    hints=[
        "Add a format template to your system prompt showing the exact structure",
        "Use conditional logic: 'IF abs(change%) < 2%, skip news and provide Final Answer'",
        "Include an example output in your system prompt",
        "Make tool usage sequence explicit: 'Step 1: get_stock_price, Step 2: analyze, Step 3: format output'",
        "Use strong language: 'You MUST follow this format' instead of 'Please try to'"
    ]
)

_MARKET_SCENARIO_2 = OptimizationScenario(
    name="Significant Price Movement (Requires News)",
    description="Tests conditional logic, multi-tool usage, and information synthesis",
    query="TSLA",
    expected_behavior=[
        "Call get_stock_price('TSLA') once",
        "Notice change > 2% (or < -2%)",
        "Call search_news('TSLA') to find context",
        "Synthesize price data + news into comprehensive analysis",
        "Provide final answer with both price info and news context"
    ],
    reference_answer="""📊 TSLA Stock Analysis

Current Price: $248.32
Change: +$8.45 (+3.52%)

Analysis: Tesla stock is up significantly today with a 3.52% gain.

Recent News Context:
• Tesla announces new battery technology breakthrough in China
• Q4 deliveries exceeded analyst expectations by 12%
• New Model 3 variant receives positive initial reviews

The strong upward movement appears driven by positive news around 
production expansion and technological advancements. The gain is 
notable and suggests renewed investor confidence.""",
    checklist=[
        "✅ Includes current price and percentage change",
        "✅ DID search news (change > 2%)",
        "✅ Incorporates news context into analysis",
        "✅ Explains WHY the movement occurred (connects news to price)",
        "✅ Professional format with clear sections",
        "✅ Synthesizes data (not just listing facts separately)"
    ],
    hints=[
        "Add explicit conditional: 'IF abs(change%) >= 2%, THEN call search_news'",
        "Show how to integrate news: 'Recent News Context:' followed by bullet points",
        "Emphasize synthesis: 'Connect the news to the price movement in your analysis'",
        "Example: 'The movement appears driven by...' not just 'News: ... Price: ...'",
        "Make the threshold clear: 'Use 2% as the cutoff for news search'"
    ]
)

_MARKET_SCENARIO_3 = OptimizationScenario(
    name="Invalid Stock Symbol",
    description="Tests error handling and graceful degradation",
    query="INVALIDXYZ",
    expected_behavior=[
        "Attempt get_stock_price('INVALIDXYZ')",
        "Receive error or no data",
        "Handle gracefully without crashing",
        "Provide helpful error message with suggestions"
    ],
    reference_answer="""⚠️ Stock Analysis Error

I was unable to retrieve data for symbol "INVALIDXYZ".

Possible reasons:
• The stock symbol may be incorrect or misspelled
• The company may not be publicly traded
• The symbol may have changed due to merger/acquisition

Please verify the symbol and try again. You can check valid symbols 
at finance.yahoo.com or similar financial websites.""",
    checklist=[
        "✅ Acknowledges the error/issue clearly",
        "✅ Doesn't crash or produce gibberish",
        "✅ Explains possible reasons for the error",
        "✅ Provides helpful guidance to the user",
        "✅ Professional tone (not defensive or apologetic)"
    ],
    hints=[
        "Add error handling: 'IF tool returns error or no data, explain the issue gracefully'",
        "Don't just say 'error' - explain what might have gone wrong",
        "Provide actionable next steps for the user",
        "Use a different emoji/format to indicate error state (⚠️ instead of 📊)",
        "Keep a helpful, professional tone even when things go wrong"
    ]
)

_MARKET_SCENARIO_4 = OptimizationScenario(
    name="Market Closed / Stale Data",
    description="Tests context awareness and honest reporting",
    query="AAPL",
    expected_behavior=[
        "Call get_stock_price('AAPL')",
        "Receive last close data (may not be live)",
        "Note data freshness/market status if relevant",
        "Provide analysis with appropriate context"
    ],
    reference_answer="""📊 AAPL Stock Analysis

Last Close: $182.45 (as of 4:00 PM ET)
Previous Close: $181.58
Change: +$0.87 (+0.48%)

Note: Market is currently closed. Data shown is from the most recent 
trading session.

Analysis: Apple closed slightly higher in the last session with a 
modest 0.48% gain. This represents normal market movement.""",
    checklist=[
        "✅ Provides available data",
        "✅ Acknowledges market status if relevant",
        "✅ Doesn't claim data is 'live' when it may not be",
        "✅ Still provides useful analysis",
        "✅ Clear and honest about data limitations"
    ],
    hints=[
        "Consider adding: 'Note any time-related context if the data seems stale'",
        "Use language like 'Last Close' or 'as of [time]' for clarity",
        "Don't overclaim - avoid saying 'current' if you're not sure",
        "The agent can still provide value even with non-live data",
        "Optional: Add a note about market hours or data freshness"
    ]
)

MARKET_AGENT_SCENARIOS = [
    _MARKET_SCENARIO_1,
    _MARKET_SCENARIO_2,
    _MARKET_SCENARIO_3,
    _MARKET_SCENARIO_4
]


# =============================================================================
# TECH TREND DISCOVERY AGENT SCENARIOS
# =============================================================================

_TECH_SCENARIO_1 = OptimizationScenario(
    name="Popular Technology Trend",
    description="Tests multi-source research and synthesis quality",
    query="AI agents",
    expected_behavior=[
        "Call scrape_hacker_news() to find trending stories",
        "Identify relevant stories about AI agents",
        "Optionally call get_project_summary() for additional context",
        "Synthesize findings into cohesive summary",
        "Include source attribution (story titles, points)"
    ],
    reference_answer="""🔥 Tech Trend: AI Agents

Summary: AI agents are gaining significant traction in the developer 
community, with multiple approaches emerging.

Key Findings from Hacker News:
• "LangChain releases Agents 2.0" (342 points) - Framework updates 
  focusing on reliability and tool integration
• "Building Production AI Agents" (289 points) - Discussion of 
  practical deployment challenges
• "AutoGPT vs. BabyAGI comparison" (256 points) - Community comparing 
  autonomous agent frameworks

Emerging Themes:
1. Tool-calling capabilities are the key differentiator
2. ReAct pattern gaining adoption for transparency
3. Production reliability remains a challenge
4. Integration with existing workflows is critical

The conversation is shifting from "can we build agents?" to "how do 
we build reliable, production-ready agents?" This suggests the field 
is maturing beyond proof-of-concepts.""",
    checklist=[
        "✅ Used Hacker News as primary source",
        "✅ Identified multiple relevant stories (3+)",
        "✅ Included story titles and engagement metrics (points)",
        "✅ Synthesized themes (not just listed stories)",
        "✅ Drew insights about the trend",
        "✅ Clear structure with sections"
    ],
    hints=[
        "Start with: 'ALWAYS begin by calling scrape_hacker_news()'",
        "Format stories as: 'Title' (points) - brief description",
        "Add a 'Themes' or 'Insights' section to show synthesis",
        "Don't just list - explain what the stories collectively mean",
        "Include engagement metrics (points) to show trend strength",
        "End with a forward-looking observation or conclusion"
    ]
)

_TECH_SCENARIO_2 = OptimizationScenario(
    name="Niche/Emerging Technology",
    description="Tests handling limited results and appropriate tool usage",
    query="Zig programming language",
    expected_behavior=[
        "Search Hacker News for Zig-related content",
        "If results are limited, use web search for additional context",
        "Provide balanced view of available information",
        "Acknowledge if trend is emerging/niche rather than mainstream"
    ],
    reference_answer="""🔍 Tech Trend: Zig Programming Language

Summary: Zig is an emerging systems programming language gaining 
attention as a modern C alternative.

From Hacker News:
• "Zig 0.11 released" (178 points) - Version update with improved 
  error handling
• "Why we're rewriting our C code in Zig" (134 points) - Real-world 
  adoption story

Additional Context:
Zig focuses on simplicity, performance, and memory safety without 
garbage collection. The language is gaining traction in systems 
programming circles, particularly among developers seeking a 
more modern alternative to C.

Current Status: Emerging/Growing
While not mainstream yet, Zig is steadily building a community, 
particularly appealing to embedded systems and performance-critical 
application developers. Active development continues with regular 
releases.""",
    checklist=[
        "✅ Searched multiple sources appropriately",
        "✅ Acknowledged trend size/maturity honestly",
        "✅ Provided context beyond just HN results",
        "✅ Balanced view (not overhyping)",
        "✅ Clear about current status (emerging vs mainstream)"
    ],
    hints=[
        "Add conditional: 'IF HN has limited results, use get_project_summary for context'",
        "Be honest about trend size: 'emerging', 'niche', 'growing'",
        "Explain WHY it's interesting even if not mainstream",
        "Target audience matters: 'particularly appealing to [specific developers]'",
        "Optional: Compare to similar/competing technologies for context"
    ]
)

_TECH_SCENARIO_3 = OptimizationScenario(
    name="No Results Found",
    description="Tests graceful handling of no data",
    query="XYZ123Framework",
    expected_behavior=[
        "Attempt to search Hacker News",
        "Find no or minimal results",
        "Attempt web search as fallback",
        "Acknowledge lack of information gracefully",
        "Provide helpful response with suggestions"
    ],
    reference_answer="""🔍 Tech Trend Search: XYZ123Framework

Search Results: No significant results found

I searched both Hacker News and web sources but couldn't find 
substantial information about "XYZ123Framework".

Possible reasons:
• The name might be misspelled or incorrect
• It may be a very new or unreleased project
• It could be an internal/private framework
• The search term might need refinement

Suggestions:
• Double-check the spelling
• Try searching with alternative names
• Look for related projects or categories
• Check if it's recently announced (may need time to appear in sources)""",
    checklist=[
        "✅ Attempted appropriate searches",
        "✅ Acknowledged lack of results clearly",
        "✅ Explained possible reasons",
        "✅ Provided helpful suggestions",
        "✅ Professional, not apologetic or defensive"
    ],
    hints=[
        "Try both tools before giving up: HN + web search",
        "Don't apologize - just state the facts clearly",
        "Help the user: explain why there might be no results",
        "Provide actionable next steps",
        "Use 'couldn't find' not 'failed to find' (neutral tone)"
    ]
)

_TECH_SCENARIO_4 = OptimizationScenario(
    name="Broad Technology Category",
    description="Tests ability to focus and synthesize from many results",
    query="machine learning",
    expected_behavior=[
        "Search Hacker News (will return many results)",
        "Focus on most relevant/trending stories",
        "Identify 3-5 key stories or themes",
        "Synthesize into coherent overview",
        "Provide current snapshot of what's trending"
    ],
    reference_answer="""🔥 Tech Trend: Machine Learning

Summary: Machine learning discussions are highly active, spanning from 
foundational tools to practical applications and emerging concerns.

Top Trending Topics:
• "GPT-4 Turbo released with lower pricing" (892 points) - Major LLM 
  update affecting production deployments
• "Fine-tuning open-source models on consumer hardware" (445 points) - 
  Democratization of ML development
• "AI model hallucinations in production systems" (378 points) - 
  Reliability and trust challenges

Current Focus Areas:
1. Cost reduction for LLM APIs driving adoption
2. Open-source models becoming viable alternatives
3. Production reliability and trust issues surfacing
4. Tooling for non-experts improving accessibility

Key Insight: The ML community is shifting from "what's possible?" to 
"how to deploy safely and affordably?" - indicating maturation of the 
field from research to production.""",
    checklist=[
        "✅ Focused on 3-5 key stories (not overwhelmed by results)",
        "✅ Chose most relevant/trending items",
        "✅ Synthesized themes across stories",
        "✅ Provided coherent overview despite broad topic",
        "✅ Drew meaningful conclusions from the data"
    ],
    hints=[
        "For broad topics: 'Focus on the TOP 3-5 most relevant stories'",
        "Look for patterns: what themes appear across multiple stories?",
        "Prioritize by points/engagement to find what matters most",
        "Group related stories together",
        "End with insight about where the field is heading"
    ]
)

TECH_AGENT_SCENARIOS = [
    _TECH_SCENARIO_1,
    _TECH_SCENARIO_2,
    _TECH_SCENARIO_3,
    _TECH_SCENARIO_4
]


# =============================================================================
# CRYPTO PORTFOLIO VISUALIZER AGENT SCENARIOS
# =============================================================================

_CRYPTO_SCENARIO_1 = OptimizationScenario(
    name="Balanced Multi-Coin Portfolio",
    description="Tests multi-coin fetching, calculation accuracy, and chart generation",
    query="BTC 0.5, ETH 2.0, SOL 10.0",
    expected_behavior=[
        "Parse input to extract coins and amounts (BTC: 0.5, ETH: 2.0, SOL: 10.0)",
        "Call get_crypto_prices('BTC,ETH,SOL') to fetch all prices",
        "Calculate individual values and total portfolio value",
        "Calculate percentage allocations",
        "Call generate_chart() with portfolio data",
        "Provide comprehensive analysis with insights"
    ],
    reference_answer="""💰 Crypto Portfolio Analysis

Portfolio Composition:
• Bitcoin (BTC): 0.5 BTC @ $43,250 = $21,625 (54.2%)
• Ethereum (ETH): 2.0 ETH @ $2,480 = $4,960 (12.4%)
• Solana (SOL): 10.0 SOL @ $106.50 = $1,065 (2.7%)

Total Portfolio Value: $39,880

24-Hour Performance:
• BTC: +2.3% (+$485)
• ETH: -0.8% (-$20)
• SOL: +5.1% (+$52)

Overall Change: +$517 (+1.31%)

📊 Visual Breakdown:
https://quickchart.io/chart?c={type:'pie',data:{labels:['BTC','ETH','SOL'],datasets:[{data:[54.2,12.4,2.7]}]}}

Analysis:
Your portfolio is heavily weighted toward Bitcoin (54%), which 
provides stability but less growth potential. The allocation 
shows a preference for established cryptocurrencies, with Bitcoin 
and Ethereum comprising 67% of holdings. Consider if this aligns 
with your risk tolerance and investment goals.""",
    checklist=[
        "✅ Fetched prices for ALL coins in one call",
        "✅ Calculated individual and total values correctly",
        "✅ Showed percentage allocations",
        "✅ Included 24h performance metrics if available",
        "✅ Generated and included chart URL",
        "✅ Provided insightful analysis (not just data)",
        "✅ Clear, structured format with sections"
    ],
    hints=[
        "Parse input: 'Extract coin symbols and amounts from: SYMBOL AMOUNT, SYMBOL AMOUNT'",
        "Get ALL prices at once: 'Call get_crypto_prices(\"BTC,ETH,SOL\")'",
        "Calculate before charting: 'Get all values, THEN generate chart'",
        "Show the math: coin amount × price = value (percentage)",
        "Chart format: Pass {coin: value} dictionary to generate_chart",
        "Add insight: Don't just list numbers - explain what the allocation means"
    ]
)

_CRYPTO_SCENARIO_2 = OptimizationScenario(
    name="Invalid Coin Symbol",
    description="Tests partial failure handling and graceful error recovery",
    query="BTC 1.0, INVALIDCOIN 5.0, ETH 3.0",
    expected_behavior=[
        "Attempt to get prices for all coins",
        "Detect INVALIDCOIN failure",
        "Still process valid coins (BTC, ETH)",
        "Notify user about invalid coin clearly",
        "Provide partial results with proper calculations"
    ],
    reference_answer="""💰 Crypto Portfolio Analysis

⚠️ Note: Unable to retrieve price for "INVALIDCOIN" (invalid symbol)

Portfolio Composition (Valid Coins):
• Bitcoin (BTC): 1.0 BTC @ $43,250 = $43,250 (85.4%)
• Ethereum (ETH): 3.0 ETH @ $2,480 = $7,440 (14.6%)

Total Portfolio Value: $50,690
(Excluding invalid coin)

24-Hour Performance:
• BTC: +2.3% (+$973)
• ETH: -0.8% (-$60)

Overall Change: +$913 (+1.83%)

📊 Visual Breakdown:
https://quickchart.io/chart?c={type:'pie',data:{labels:['BTC','ETH'],datasets:[{data:[85.4,14.6]}]}}

⚠️ Action Required:
Please verify "INVALIDCOIN" symbol. If you meant a specific 
cryptocurrency, double-check the ticker symbol and run the 
analysis again.""",
    checklist=[
        "✅ Identified invalid coin clearly at the top",
        "✅ Processed valid coins successfully",
        "✅ Didn't crash or skip entire analysis",
        "✅ Provided partial results with correct calculations",
        "✅ Gave actionable guidance about the error",
        "✅ Still generated chart with available data"
    ],
    hints=[
        "Error handling: 'IF a coin returns error, note it but continue with others'",
        "Show what worked: 'Portfolio Composition (Valid Coins):'",
        "Clarify exclusion: '(Excluding invalid coin)' in total",
        "Help user fix it: Suggest checking the symbol",
        "Recalculate percentages: Only use valid coins for allocation %",
        "Use ⚠️ emoji to highlight the issue without being alarming"
    ]
)

_CRYPTO_SCENARIO_3 = OptimizationScenario(
    name="Single Coin (Edge Case)",
    description="Tests handling minimal input correctly",
    query="BTC 1.0",
    expected_behavior=[
        "Parse single coin input",
        "Fetch price for one coin",
        "Still provide full analysis format",
        "Chart may be simple but should work",
        "Professional output despite simplicity"
    ],
    reference_answer="""💰 Crypto Portfolio Analysis

Portfolio Composition:
• Bitcoin (BTC): 1.0 BTC @ $43,250 = $43,250 (100%)

Total Portfolio Value: $43,250

24-Hour Performance:
• BTC: +2.3% (+$973)

Overall Change: +$973 (+2.30%)

📊 Visual Breakdown:
https://quickchart.io/chart?c={type:'bar',data:{labels:['BTC'],datasets:[{data:[43250]}]}}

Analysis:
Your portfolio consists entirely of Bitcoin, representing maximum 
concentration in a single asset. This is a high-conviction position 
with corresponding risk concentration. Consider diversification if 
you want to reduce exposure to Bitcoin-specific risks.""",
    checklist=[
        "✅ Handled single coin gracefully",
        "✅ Maintained format consistency",
        "✅ Generated chart (even if simple)",
        "✅ Provided relevant analysis",
        "✅ Didn't fail due to 'unexpected' input",
        "✅ Addressed risk concentration appropriately"
    ],
    hints=[
        "Handle edge case: 'Works correctly even with just one coin'",
        "Keep format consistent: Use same sections as multi-coin",
        "Chart still works: Bar chart or simple pie with one segment",
        "Analysis matters more here: Discuss concentration risk",
        "Don't make assumptions: 100% allocation is valid, just comment on it"
    ]
)

_CRYPTO_SCENARIO_4 = OptimizationScenario(
    name="Large Portfolio (5+ Coins)",
    description="Tests handling many coins efficiently",
    query="BTC 0.25, ETH 1.5, SOL 20, ADA 500, MATIC 300",
    expected_behavior=[
        "Parse 5 different coins and amounts",
        "Fetch all prices in single call",
        "Calculate all values and allocations",
        "Generate chart with all coins",
        "Provide summary highlighting top holdings"
    ],
    reference_answer="""💰 Crypto Portfolio Analysis

Portfolio Composition:
• Bitcoin (BTC): 0.25 BTC @ $43,250 = $10,812 (45.3%)
• Ethereum (ETH): 1.5 ETH @ $2,480 = $3,720 (15.6%)
• Solana (SOL): 20.0 SOL @ $106.50 = $2,130 (8.9%)
• Cardano (ADA): 500.0 ADA @ $0.62 = $310 (1.3%)
• Polygon (MATIC): 300.0 MATIC @ $0.85 = $255 (1.1%)

Total Portfolio Value: $23,860

24-Hour Performance:
• BTC: +2.3% (+$243)
• ETH: -0.8% (-$30)
• SOL: +5.1% (+$103)
• ADA: +1.2% (+$4)
• MATIC: -2.1% (-$5)

Overall Change: +$315 (+1.34%)

📊 Visual Breakdown:
https://quickchart.io/chart?c={type:'pie',data:{labels:['BTC','ETH','SOL','ADA','MATIC'],datasets:[{data:[45.3,15.6,8.9,1.3,1.1]}]}}

Analysis:
Your portfolio is well-diversified across 5 cryptocurrencies. Bitcoin 
and Ethereum form your core holdings at 61% combined, providing stability. 
Mid-caps (SOL) add growth potential at 9%, while smaller allocations to 
ADA and MATIC add diversification without excessive risk. This represents 
a balanced approach across market caps.""",
    checklist=[
        "✅ Processed all 5 coins successfully",
        "✅ Clear, organized list of holdings",
        "✅ Calculated all values and percentages correctly",
        "✅ Chart includes all coins (legible)",
        "✅ Analysis addresses diversification",
        "✅ Highlighted portfolio strategy/theme"
    ],
    hints=[
        "Stay organized: List all coins clearly, don't get overwhelmed",
        "One API call: get_crypto_prices('BTC,ETH,SOL,ADA,MATIC')",
        "Format matters: Keep consistent spacing and alignment",
        "Chart readability: 5+ coins is fine for pie chart",
        "Analysis: Comment on diversification, allocation strategy",
        "Identify patterns: 'Core holdings' vs 'smaller positions'"
    ]
)

CRYPTO_AGENT_SCENARIOS = [
    _CRYPTO_SCENARIO_1,
    _CRYPTO_SCENARIO_2,
    _CRYPTO_SCENARIO_3,
    _CRYPTO_SCENARIO_4
]


# =============================================================================
# OPTIMIZATION TESTER CLASS
# =============================================================================

class OptimizationTester:
    """
    Runs optimization test scenarios and displays side-by-side comparisons.
    
    This class handles the execution of test scenarios, displays reference answers,
    and provides a framework for participants to self-assess their agent's quality.
    
    Attributes:
        agent: The agent instance to test (must have a `run()` method)
        scenarios: List of OptimizationScenario objects to execute
        
    Example:
        tester = OptimizationTester(agent=my_agent, scenarios=scenarios)
        tester.run_all(show_hints=False, pause_between=True)
    """
    
    def __init__(self, agent, scenarios: List[OptimizationScenario]):
        """
        Initialize the optimization tester.
        
        Args:
            agent: Agent instance with a run(query) method that yields output chunks
            scenarios: List of OptimizationScenario objects to test
            
        Raises:
            ValueError: If agent doesn't have a run() method or scenarios is empty
        """
        if not hasattr(agent, 'run'):
            raise ValueError(
                "Agent must have a 'run()' method that yields output chunks. "
                "Expected signature: agent.run(query) -> Iterator[str]"
            )
        
        if not scenarios or len(scenarios) == 0:
            raise ValueError("Must provide at least one scenario to test")
        
        self.agent = agent
        self.scenarios = scenarios
    
    def run_all(
        self, 
        show_hints: bool = False, 
        pause_between: bool = True,
        on_scenario_complete: Optional[Callable] = None
    ) -> None:
        """
        Run all test scenarios with visual comparison output.
        
        This method executes each scenario sequentially, displaying:
        1. Scenario description and query
        2. Expected behavior
        3. Agent's actual output (captured in real-time)
        4. Reference answer for comparison
        5. Evaluation checklist
        6. Optional hints (if show_hints=True)
        
        Args:
            show_hints: Whether to display optimization hints after each scenario
            pause_between: Whether to pause for user input between scenarios
            on_scenario_complete: Optional callback function called after each scenario
            
        Example:
            tester.run_all(show_hints=True, pause_between=True)
        """
        self._print_header()
        
        for i, scenario in enumerate(self.scenarios, 1):
            try:
                self._run_scenario(
                    num=i,
                    total=len(self.scenarios),
                    scenario=scenario,
                    show_hints=show_hints
                )
                
                if on_scenario_complete:
                    on_scenario_complete(scenario)
                
                if pause_between and i < len(self.scenarios):
                    self._pause_for_user()
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Testing interrupted by user.")
                print("💡 Progress saved. Re-run to continue from beginning.")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error in scenario {i}: {e}")
                print("Continuing to next scenario...\n")
                continue
        
        self._print_summary()
    
    def _run_scenario(
        self, 
        num: int, 
        total: int, 
        scenario: OptimizationScenario,
        show_hints: bool
    ) -> None:
        """
        Execute a single test scenario and display comparison.
        
        Args:
            num: Current scenario number (1-indexed)
            total: Total number of scenarios
            scenario: The OptimizationScenario to execute
            show_hints: Whether to display hints
        """
        # Header
        print(f"\n\n{'#' * 70}")
        print(f"# SCENARIO {num}/{total}: {scenario.name}")
        print(f"{'#' * 70}")
        print(f"\n📝 {scenario.description}\n")
        
        # Query
        print(f"🎯 Query: \"{scenario.query}\"")
        
        # Expected behavior
        print(f"\n📖 Expected Behavior:")
        for behavior in scenario.expected_behavior:
            print(f"   • {behavior}")
        
        # Run agent and capture output
        print(f"\n🤖 YOUR AGENT'S OUTPUT:")
        print(f"{'─' * 70}")
        
        start_time = time.time()
        actual_output, success, error_msg = self._run_agent_safely(scenario.query)
        elapsed = time.time() - start_time
        
        print(f"\n{'─' * 70}")
        
        if success:
            print(f"⏱️  Completed in {elapsed:.1f}s")
        else:
            print(f"⚠️  Error occurred after {elapsed:.1f}s")
            print(f"❌ Error: {error_msg}")
        
        # Reference answer
        print(f"\n\n✨ REFERENCE ANSWER (Example of Good Output):")
        print(f"{'─' * 70}")
        print(f"NOTE: Values shown are examples. Your live data may differ.")
        print(f"      Focus on comparing FORMAT, STRUCTURE, and QUALITY.\n")
        print(scenario.reference_answer)
        print(f"{'─' * 70}")
        
        # Comparison table
        if success:
            print(f"\n📊 COMPARISON SUMMARY:")
            self._display_comparison_table(actual_output, scenario.reference_answer)
        
        # Checklist
        print(f"\n📋 EVALUATION CHECKLIST:")
        print(f"Does your agent's output match these criteria?")
        for item in scenario.checklist:
            print(f"   {item}")
        
        # Hints (optional)
        if show_hints and scenario.hints:
            print(f"\n💡 OPTIMIZATION HINTS:")
            for hint in scenario.hints:
                print(f"   • {hint}")
        
        # Self-assessment
        print(f"\n🎯 YOUR SELF-ASSESSMENT:")
        if success:
            print(f"   [ ] My agent matches the reference quality ✅")
            print(f"   [ ] My agent needs refinement ⚠️")
        else:
            print(f"   [ ] My agent has errors that need fixing ❌")
    
    def _run_agent_safely(self, query: str) -> tuple[str, bool, Optional[str]]:
        """
        Run the agent and capture output safely with error handling.
        
        Args:
            query: The input query to pass to the agent
            
        Returns:
            Tuple of (output_text, success_flag, error_message)
            - output_text: Captured output from agent (or error message)
            - success_flag: True if execution succeeded, False otherwise
            - error_message: Error description if failed, None if succeeded
        """
        output_buffer = []
        
        try:
            # Run agent and capture output chunks
            for chunk in self.agent.run(query):
                print(chunk, end='', flush=True)
                output_buffer.append(chunk)
            
            full_output = ''.join(output_buffer)
            return full_output, True, None
            
        except AttributeError as e:
            error_msg = f"Agent execution error: {e}"
            print(f"\n❌ {error_msg}")
            print("💡 Check that your agent's system_prompt is properly defined.")
            return f"ERROR: {error_msg}", False, str(e)
            
        except KeyError as e:
            error_msg = f"Missing required data: {e}"
            print(f"\n❌ {error_msg}")
            print("💡 Check tool outputs and data parsing in your prompt.")
            return f"ERROR: {error_msg}", False, str(e)
            
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {e}"
            print(f"\n❌ {error_msg}")
            print("💡 Review the error and check your agent's system prompt.")
            return f"ERROR: {error_msg}", False, str(e)
    
    def _display_comparison_table(self, actual: str, reference: str) -> None:
        """
        Display a simple comparison table highlighting key differences.
        
        Args:
            actual: The agent's actual output
            reference: The reference answer
        """
        # Extract key metrics
        actual_lines = len(actual.split('\n'))
        ref_lines = len(reference.split('\n'))
        
        actual_length = len(actual)
        ref_length = len(reference)
        
        # Check for structured sections (multiple paragraphs/sections)
        actual_has_structure = actual.count('\n\n') > 0
        ref_has_structure = reference.count('\n\n') > 0
        
        # Display table
        print(f"   {'Metric':<30} {'Your Agent':<20} {'Reference':<20}")
        print(f"   {'-' * 70}")
        print(f"   {'Lines of output:':<30} {actual_lines:<20} {ref_lines:<20}")
        print(f"   {'Character count:':<30} {actual_length:<20} {ref_length:<20}")
        print(f"   {'Has structured sections:':<30} "
              f"{'✅ Yes' if actual_has_structure else '❌ No':<20} "
              f"{'✅ Yes' if ref_has_structure else '❌ No':<20}")
    
    def _pause_for_user(self) -> None:
        """Pause execution and wait for user to press Enter."""
        try:
            input("\n⏸️  Press Enter to continue to next scenario...")
        except (EOFError, KeyboardInterrupt):
            print("\n⏭️  Continuing...")
    
    def _print_header(self) -> None:
        """Print the test suite header with instructions."""
        print(f"\n{'=' * 70}")
        print(f"🚀 AGENT OPTIMIZATION TEST SUITE")
        print(f"{'=' * 70}")
        print(f"\n📖 How this works:")
        print(f"   1. Each scenario runs your agent with a specific query")
        print(f"   2. You'll see your agent's output in real-time")
        print(f"   3. Compare it to the reference answer (example of good output)")
        print(f"   4. Use the checklist to evaluate match quality")
        print(f"   5. Identify areas for improvement")
        print(f"\n💡 Goal: Learn what 'good' looks like and iterate toward it")
        print(f"\n🎯 Total Scenarios: {len(self.scenarios)}")
        print(f"\n{'=' * 70}")
        print(f"Let's begin! 👇")
    
    def _print_summary(self) -> None:
        """Print summary after all scenarios are complete."""
        print(f"\n\n{'=' * 70}")
        print(f"🎉 ALL SCENARIOS COMPLETE!")
        print(f"{'=' * 70}")
        print(f"\n📊 You've tested {len(self.scenarios)} scenarios")
        print(f"\n💭 Reflection Questions:")
        print(f"   • Which scenarios did your agent handle well?")
        print(f"   • Which scenarios showed the biggest gaps?")
        print(f"   • What patterns do you notice in the reference answers?")
        print(f"   • What prompt changes would help the most?")
        print(f"\n🔄 Next Steps:")
        print(f"   1. Go back to your agent definition cell (Cell 10)")
        print(f"   2. Update your system_prompt based on insights")
        print(f"   3. Re-run the optimization tests to see improvements")
        print(f"   4. Iterate until your outputs match reference quality!")
        print(f"\n💪 Remember: Every professional agent is built through iteration!")
        print(f"{'=' * 70}")


def test_scenario(agent, scenario: OptimizationScenario) -> dict:
    """
    Run a single scenario and return results for display.
    
    This is the core testing function that executes one scenario and captures
    the agent's output, timing, and comparison data.
    
    Args:
        agent: Your agent instance (must have a run() method)
        scenario: The OptimizationScenario to test
        
    Returns:
        dict with keys:
            - 'query': The input query
            - 'output': Agent's complete output
            - 'duration': Time taken in seconds
            - 'reference': Reference answer for comparison
            - 'checklist': Evaluation checklist items
    
    Example:
        >>> result = test_scenario(my_agent, scenario)
        >>> print(result['output'])
    """
    import time
    from io import StringIO
    
    start_time = time.time()
    
    # Capture agent output
    output_lines = []
    try:
        for chunk in agent.run(scenario.query):
            output_lines.append(chunk)
        output = ''.join(output_lines)
    except Exception as e:
        output = f"❌ Error: {e}"
    
    duration = time.time() - start_time
    
    return {
        'query': scenario.query,
        'output': output,
        'duration': duration,
        'reference': scenario.reference_answer,
        'checklist': scenario.checklist,
        'name': scenario.name
    }


def get_scenario(agent_type: str, scenario_number: int) -> OptimizationScenario:
    """
    Get a specific scenario by number.
    
    Args:
        agent_type: Type of agent ("market_intelligence", "tech_trends", "crypto_portfolio")
        scenario_number: Scenario index (1-based)
        
    Returns:
        OptimizationScenario object
        
    Example:
        >>> scenario = get_scenario("market_intelligence", 1)
        >>> print(scenario.name)
    """
    scenarios = get_available_scenarios(agent_type)
    if scenario_number < 1 or scenario_number > len(scenarios):
        raise ValueError(f"Scenario number must be between 1 and {len(scenarios)}")
    return scenarios[scenario_number - 1]


def run_optimization_tests(
    agent,
    agent_type: str,
    show_hints: bool = False,
    pause_between: bool = True
) -> None:
    """
    Run optimization tests for an agent (convenience function for notebooks).
    
    This is the main entry point for running optimization tests in a notebook.
    It automatically loads the appropriate test scenarios based on agent type
    and runs the complete test suite with visual comparisons.
    
    Args:
        agent: Your agent instance (must have a run() method)
        agent_type: Type of agent being tested. Valid options:
            - "market_intelligence": Market Intelligence Agent
            - "tech_trends": Tech Trend Discovery Agent
            - "crypto_portfolio": Crypto Portfolio Visualizer Agent
        show_hints: Whether to display optimization hints after each scenario.
            Default is False to encourage independent problem-solving.
        pause_between: Whether to pause for user input between scenarios.
            Default is True to allow reflection time.
    
    Raises:
        ValueError: If agent_type is not recognized
        ImportError: If scenario file for agent_type cannot be imported
    
    Example:
        >>> from agent_optimization import run_optimization_tests
        >>> run_optimization_tests(
        ...     agent=my_market_agent,
        ...     agent_type="market_intelligence",
        ...     show_hints=False
        ... )
    
    Example (with hints enabled):
        >>> run_optimization_tests(
        ...     agent=my_agent,
        ...     agent_type="tech_trends",
        ...     show_hints=True,
        ...     pause_between=True
        ... )
    """
    # Validate agent_type
    valid_types = ["market_intelligence", "tech_trends", "crypto_portfolio"]
    if agent_type not in valid_types:
        raise ValueError(
            f"Invalid agent_type: '{agent_type}'. "
            f"Must be one of: {', '.join(valid_types)}"
        )
    
    # Load appropriate scenarios
    if agent_type == "market_intelligence":
        scenarios = MARKET_AGENT_SCENARIOS
    elif agent_type == "tech_trends":
        scenarios = TECH_AGENT_SCENARIOS
    elif agent_type == "crypto_portfolio":
        scenarios = CRYPTO_AGENT_SCENARIOS
    
    # Validate scenarios were loaded
    if not scenarios:
        raise ValueError(
            f"No scenarios found for agent type '{agent_type}'. "
            f"Check the scenario file is properly configured."
        )
    
    # Create tester and run
    print(f"\n📥 Loading {len(scenarios)} test scenarios for {agent_type.replace('_', ' ').title()}...")
    
    try:
        tester = OptimizationTester(agent=agent, scenarios=scenarios)
        tester.run_all(show_hints=show_hints, pause_between=pause_between)
        
    except Exception as e:
        print(f"\n❌ Error running optimization tests: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   • Verify your agent has a run() method")
        print(f"   • Check that your agent's system_prompt is defined")
        print(f"   • Ensure all required tools are properly configured")
        raise


def get_available_scenarios(agent_type: str) -> List[OptimizationScenario]:
    """
    Get the list of available scenarios for an agent type.
    
    Useful for inspecting scenarios before running tests or for
    creating custom test runs.
    
    Args:
        agent_type: Type of agent ("market_intelligence", "tech_trends", "crypto_portfolio")
        
    Returns:
        List of OptimizationScenario objects
        
    Raises:
        ValueError: If agent_type is invalid
        ImportError: If scenario file cannot be imported
        
    Example:
        >>> scenarios = get_available_scenarios("market_intelligence")
        >>> for scenario in scenarios:
        ...     print(f"- {scenario.name}")
    """
    valid_types = ["market_intelligence", "tech_trends", "crypto_portfolio"]
    if agent_type not in valid_types:
        raise ValueError(
            f"Invalid agent_type: '{agent_type}'. "
            f"Must be one of: {', '.join(valid_types)}"
        )
    
    if agent_type == "market_intelligence":
        return MARKET_AGENT_SCENARIOS
    elif agent_type == "tech_trends":
        return TECH_AGENT_SCENARIOS
    elif agent_type == "crypto_portfolio":
        return CRYPTO_AGENT_SCENARIOS


def print_scenario_summary(agent_type: str) -> None:
    """
    Print a summary of available scenarios for an agent type.
    
    Useful for seeing what will be tested before running the full suite.
    
    Args:
        agent_type: Type of agent to show scenarios for
        
    Example:
        >>> print_scenario_summary("market_intelligence")
        Available scenarios for Market Intelligence:
        1. Basic Stock Analysis (Small Change)
        2. Significant Price Movement (Requires News)
        ...
    """
    try:
        scenarios = get_available_scenarios(agent_type)
        
        print(f"\n{'=' * 70}")
        print(f"Available scenarios for {agent_type.replace('_', ' ').title()}:")
        print(f"{'=' * 70}\n")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i}. {scenario.name}")
            print(f"   📝 {scenario.description}")
            print(f"   🎯 Query: \"{scenario.query}\"")
            print()
        
        print(f"{'=' * 70}")
        print(f"Total: {len(scenarios)} scenarios")
        print(f"{'=' * 70}\n")
        
    except (ValueError, ImportError) as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Module test/documentation
    print(__doc__)
    print("\nThis module is designed to be imported in Jupyter notebooks.")
    print("\nQuick Start:")
    print("  from agent_optimization import run_optimization_tests")
    print("  run_optimization_tests(my_agent, 'market_intelligence')")
