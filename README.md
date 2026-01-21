# <p align="center"> Python: Pricing Model <p/>
<br>**Nattawut Boonnoon**<br/>
- LinkedIn: www.linkedin.com/in/nattawut-bn
- Email: nattawut.boonnoon@hotmail.com
- Phone: (+66) 92 271 6680

***Overview***
- 
Link: <br><br/>
This is my options pricing calculator that combines educational clarity with real-world utility. It uses the Black-Scholes model to price European options and includes advanced features like implied volatility calculation, strategy analysis, and live market data integration. Built with Python and Streamlit, it serves both learning and practical analysis.

***Black-Scholes-Merton Model***
-
The Black-Scholes, or Black-Scholes-Merton model is a mathematical model that describes the trends of a financial market, including derivative investment instruments. The formula and model are named after the economists Fischer Black and Myron Scholes. Occasionally, attribution is also awarded to Robert C. Merton, who was the first to write an academic paper on the topic.

The model's fundamental objective is to hedge the option by purchasing and selling the underlying asset in a precise pattern to remove risk. This type of hedging is known as "constantly modified delta hedging" and forms the foundation of more complex hedging strategies utilized by investment firms and hedge funds.

Call Options Price:
`````bash
C = S₀·N(d₁) - K·e^(-rT)·N(d₂)
`````
Put Options Price:
`````bash
P = K·e^(-rT)·N(-d₂) - S₀·N(-d₁)
`````
Where:
`````bash
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
`````
Parameters:
`````bash
S₀ = Current stock price
K = Strike price
T = Time to expiration (years)
r = Risk-free interest rate
σ = Volatility (annual)
N(x) = Cumulative normal distribution
`````

***Dependencies***
-
- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `numpy` - Numerical calculations
- `scipy` - Statistical functions
- `plotly` - Interactive charts
- `matplotlib` - Additional plotting
- `seaborn` - Statistical visualizations
