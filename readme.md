**NASA Kepler Exoplanet Classifier**
*Quantifying the Transition Boundary Between Dual Entities (Planetary Candidates and False Positives): A Machine-Learning-Driven Approach to Statistical Validation in Transit Photometry of Host Stars Accounting Critical Stellar Variables*


***Overview:***
 Can machine learning help us distinguish potential exoplanets from 'false positives' using NASA and CalTech's elaborate archive recorded by the Kepler mission?

~ An exoplanet is a planet that orbits a star (or a system of binary stars) beyond our Solar System. Finding one is not as simple as pointing at a star and seeing the planet directly. Astronomers often look for subtle changes in the brightness of the host star that may indicate a planet passing in front of it. However, similar effects and signals maybe observed by inexplicable phenomena or measurement issues. This makes identifying genuine planets a challenging classification problem.

~ In order to eradicate maximum errors in measurement, this project employs machine learning to learn patterns from previous observations and use them to predict whether an object is recorded as a planet candidate or a false positive. 

***Astrophysical Background:***
~ The Kepler telescope searched for signs of exoplanets through the transit method. When an object passes in front of its host star, the observed stellar brightness may decrease periodically. The depth and duration of this transit, together with other measurements, help astronomers investigate the object's properties.

Under simplified conditions, transit depth is approximately related to the planet-to-star radius ratio by:

$$
\delta \approx \left(\frac{R_p}{R_*}\right)^2
$$
