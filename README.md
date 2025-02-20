# Commute Impact Analysis Streamlit App

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][(https://github.com/jgregg-cresa)]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<a id="readme-top"></a>

## About The Project

The **Commute Impact Analysis Streamlit App** is designed to analyze employee commute times to potential office locations using Google Maps Distance Matrix API. The app compares driving and public transit commute durations, visualizes data on an interactive map, and categorizes commute time changes into intuitive buckets.

Key Features:
* Analyze commute times for different transit methods
* Compare commute times to multiple destination locations
* Interactive map with Folium integration
* Export categorized commute data as CSV

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
* [Pandas](https://pandas.pydata.org/)
* [Folium](https://python-visualization.github.io/folium/)
* [Google Maps API](https://developers.google.com/maps/documentation)
* [TimezoneFinder](https://timezonefinder.readthedocs.io/en/latest/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

- Python 3.8+
- Google Maps API key
- ZIP code population-weighted centroids CSV file

### Installation

1. Clone the repo
   ```bash
   git clone https://github.com/your_username/CommuteImpactAnalysis.git
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Configure Google Maps API key in `secrets.toml`
   ```toml
   [google_maps]
   api_key = "YOUR_API_KEY"
   ```
5. Ensure the ZIP code data file is located in the root directory

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage

1. Run the app
   ```bash
   streamlit run StreamlitApp_PublicTransit.py
   ```
2. Upload origins and destinations CSV files in the sidebar
3. Choose transit method (driving or transit)
4. Click **Run Analysis** to generate results
5. Download the categorized data as a CSV file

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Roadmap

- [x] Analyze commute times for driving and public transit
- [x] Interactive map with employee origins and office destinations
- [ ] Multi-language support

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repo
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## License

Distributed under the MIT License. See `LICENSE.txt` for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Acknowledgments

* [Streamlit](https://streamlit.io/)
* [Google Maps API](https://developers.google.com/maps/documentation)
* [Folium](https://python-visualization.github.io/folium/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/your_username/CommuteImpactAnalysis.svg?style=for-the-badge
[contributors-url]: https://github.com/your_username/CommuteImpactAnalysis/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/your_username/CommuteImpactAnalysis.svg?style=for-the-badge
[forks-url]: https://github.com/your_username/CommuteImpactAnalysis/network/members
[stars-shield]: https://img.shields.io/github/stars/your_username/CommuteImpactAnalysis.svg?style=for-the-badge
[stars-url]: https://github.com/your_username/CommuteImpactAnalysis/stargazers
[issues-shield]: https://img.shields.io/github/issues/your_username/CommuteImpactAnalysis.svg?style=for-the-badge
[issues-url]: https://github.com/your_username/CommuteImpactAnalysis/issues
[license-shield]: https://img.shields.io/github/license/your_username/CommuteImpactAnalysis.svg?style=for-the-badge
[license-url]: https://github.com/your_username/CommuteImpactAnalysis/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/your_linkedin

