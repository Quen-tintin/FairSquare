"""
OpenStreetMap Feature Engineering
===================================
Génère des features géospatiales avancées à partir de coordonnées Lat/Long
via l'API Overpass (OSM).

Stratégie : UNE seule requête Overpass par point (évite le rate-limit 429).
Tous les POI sont récupérés en un seul appel, puis traités localement.

Features produites par point :
  - dist_metro_m          : distance au métro/RER le plus proche (m)
  - dist_park_m           : distance au parc le plus proche (m)
  - dist_school_m         : distance à l'école la plus proche (m)
  - dist_supermarket_m    : distance au supermarché le plus proche (m)
  - transit_count_500m    : nombre de stations dans 500 m
  - park_area_500m_m2     : surface totale de parcs dans 500 m (m²)
  - walkability_score     : score composite 0–100
  - dist_centre_paris_km  : distance au Châtelet (km) — pas d'API
  - commerce_count_300m   : restaurants, bars, cafés, boulangeries dans 300 m
  - transport_count_300m  : tous arrêts de transport dans 300 m
  - school_count_500m     : écoles dans 500 m
  - dist_nuisance_m       : distance à la nuisance la plus proche (station-service, etc.)
  - heritage_count_500m   : monuments / sites historiques dans 500 m

Usage:
    extractor = OSMFeatureExtractor()
    features = extractor.get_features(lat=48.8566, lon=2.3522)
    print(features.to_dict())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_REQUEST_TIMEOUT = 45.0
_COURTESY_DELAY = 5.0   # secondes entre points (politique Overpass)

_DEFAULT_RADIUS = 1_000  # mètres

# Châtelet — centre géographique de Paris pour dist_centre_paris_km
_CHATELET_LAT = 48.8566
_CHATELET_LON = 2.3471


# ------------------------------------------------------------------ #
#  Structure de données                                                #
# ------------------------------------------------------------------ #

@dataclass
class OSMFeatures:
    lat: float
    lon: float
    dist_metro_m: float | None = None
    dist_park_m: float | None = None
    dist_school_m: float | None = None
    dist_supermarket_m: float | None = None
    transit_count_500m: int = 0
    park_area_500m_m2: float = 0.0
    walkability_score: float | None = None
    # v2 features
    dist_centre_paris_km: float = 0.0
    commerce_count_300m: int = 0
    transport_count_300m: int = 0
    school_count_500m: int = 0
    dist_nuisance_m: float | None = None
    heritage_count_500m: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "dist_metro_m": self.dist_metro_m,
            "dist_park_m": self.dist_park_m,
            "dist_school_m": self.dist_school_m,
            "dist_supermarket_m": self.dist_supermarket_m,
            "transit_count_500m": self.transit_count_500m,
            "park_area_500m_m2": self.park_area_500m_m2,
            "walkability_score": self.walkability_score,
            "dist_centre_paris_km": self.dist_centre_paris_km,
            "commerce_count_300m": self.commerce_count_300m,
            "transport_count_300m": self.transport_count_300m,
            "school_count_500m": self.school_count_500m,
            "dist_nuisance_m": self.dist_nuisance_m,
            "heritage_count_500m": self.heritage_count_500m,
        }


# ------------------------------------------------------------------ #
#  Extracteur principal                                                #
# ------------------------------------------------------------------ #

class OSMFeatureExtractor:
    """
    Génère des features géospatiales OSM pour un point géographique.
    Une seule requête Overpass par point = pas de rate-limit.
    """

    def __init__(
        self,
        radius_m: int = _DEFAULT_RADIUS,
        timeout: float = _REQUEST_TIMEOUT,
        courtesy_delay: float = _COURTESY_DELAY,
    ) -> None:
        self.radius_m = radius_m
        self.timeout = timeout
        self.courtesy_delay = courtesy_delay

    def get_features(self, lat: float, lon: float) -> OSMFeatures:
        """
        Calcule toutes les features OSM en une seule requête Overpass.
        """
        features = OSMFeatures(lat=lat, lon=lon)

        # Distance to Châtelet — no API call needed
        features.dist_centre_paris_km = round(
            _haversine(lat, lon, _CHATELET_LAT, _CHATELET_LON) / 1000, 3
        )

        elements = self._fetch_all_pois(lat, lon, features)

        if elements:
            self._compute_features(lat, lon, elements, features)

        features.walkability_score = _compute_walkability(features)
        return features

    def get_features_batch(
        self, coordinates: list[tuple[float, float]]
    ) -> list[OSMFeatures]:
        """Calcule les features pour une liste de (lat, lon)."""
        results = []
        total = len(coordinates)
        for i, (lat, lon) in enumerate(coordinates):
            logger.info("OSM features — point %d/%d (%.4f, %.4f)", i + 1, total, lat, lon)
            results.append(self.get_features(lat, lon))
            if i < total - 1:
                time.sleep(self.courtesy_delay)
        return results

    # ------------------------------------------------------------------ #
    #  Requête Overpass unique                                            #
    # ------------------------------------------------------------------ #

    def _fetch_all_pois(
        self, lat: float, lon: float, features: OSMFeatures
    ) -> list[dict[str, Any]]:
        """
        Une seule requête Overpass récupère tous les POI utiles :
        transport, parcs, écoles, commerces.
        """
        r = self.radius_m
        query = f"""
        [out:json][timeout:40];
        (
          node["railway"~"station|subway_entrance"](around:{r},{lat},{lon});
          node["station"="subway"](around:{r},{lat},{lon});
          node["public_transport"="stop_position"]["train"="yes"](around:{r},{lat},{lon});
          node["public_transport"="stop_position"]["subway"="yes"](around:{r},{lat},{lon});
          node["public_transport"="stop_position"]["tram"="yes"](around:{r},{lat},{lon});
          node["highway"="bus_stop"](around:500,{lat},{lon});
          way["leisure"="park"](around:{r},{lat},{lon});
          node["amenity"~"school|college|university"](around:{r},{lat},{lon});
          way["amenity"~"school|college|university"](around:{r},{lat},{lon});
          node["shop"~"supermarket|convenience|grocery"](around:{r},{lat},{lon});
          node["amenity"~"restaurant|bar|cafe"](around:300,{lat},{lon});
          node["shop"~"bakery|pastry"](around:300,{lat},{lon});
          node["historic"](around:500,{lat},{lon});
          way["historic"](around:500,{lat},{lon});
          node["amenity"="fuel"](around:{r},{lat},{lon});
        );
        out center;
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(_OVERPASS_URL, data={"data": query})
                resp.raise_for_status()
                return resp.json().get("elements", [])
        except Exception as exc:
            msg = f"Overpass query failed: {exc}"
            logger.warning(msg)
            features.errors.append(msg)
            return []

    def _compute_features(
        self,
        lat: float,
        lon: float,
        elements: list[dict[str, Any]],
        features: OSMFeatures,
    ) -> None:
        """Trie les éléments récupérés et calcule chaque feature."""
        transit_nodes: list[dict] = []
        park_elements: list[dict] = []
        school_nodes: list[dict] = []
        market_nodes: list[dict] = []
        commerce_nodes: list[dict] = []
        heritage_nodes: list[dict] = []
        nuisance_nodes: list[dict] = []

        for e in elements:
            tags = e.get("tags", {})

            is_transit = (
                tags.get("railway") in ("station", "subway_entrance")
                or tags.get("station") == "subway"
                or tags.get("public_transport") == "stop_position"
                or tags.get("highway") == "bus_stop"
            )
            is_park = tags.get("leisure") == "park"
            is_school = tags.get("amenity") in ("school", "college", "university")
            is_market = tags.get("shop") in ("supermarket", "convenience", "grocery")
            is_commerce = (
                tags.get("amenity") in ("restaurant", "bar", "cafe")
                or tags.get("shop") in ("bakery", "pastry")
            )
            is_heritage = "historic" in tags
            is_nuisance = tags.get("amenity") == "fuel"

            if is_transit:
                transit_nodes.append(e)
            if is_park:
                park_elements.append(e)
            if is_school:
                school_nodes.append(e)
            if is_market:
                market_nodes.append(e)
            if is_commerce:
                commerce_nodes.append(e)
            if is_heritage:
                heritage_nodes.append(e)
            if is_nuisance:
                nuisance_nodes.append(e)

        # Distances au plus proche
        features.dist_metro_m = _nearest_distance(lat, lon, transit_nodes)
        features.dist_park_m = _nearest_distance(lat, lon, _extract_centers(park_elements))
        features.dist_school_m = _nearest_distance(lat, lon, _extract_centers(school_nodes))
        features.dist_supermarket_m = _nearest_distance(lat, lon, market_nodes)
        features.dist_nuisance_m = _nearest_distance(lat, lon, _extract_centers(nuisance_nodes))

        # Comptage transports dans 500 m et 300 m
        features.transit_count_500m = sum(
            1 for n in transit_nodes
            if "lat" in n and _haversine(lat, lon, n["lat"], n["lon"]) <= 500
        )
        features.transport_count_300m = sum(
            1 for n in transit_nodes
            if "lat" in n and _haversine(lat, lon, n["lat"], n["lon"]) <= 300
        )

        # Comptage commerces dans 300 m
        features.commerce_count_300m = sum(
            1 for n in commerce_nodes
            if "lat" in n and _haversine(lat, lon, n["lat"], n["lon"]) <= 300
        )

        # Comptage écoles dans 500 m
        school_centers = _extract_centers(school_nodes)
        features.school_count_500m = sum(
            1 for n in school_centers
            if _haversine(lat, lon, n["lat"], n["lon"]) <= 500
        )

        # Comptage patrimoine dans 500 m
        heritage_centers = _extract_centers(heritage_nodes)
        features.heritage_count_500m = sum(
            1 for n in heritage_centers
            if _haversine(lat, lon, n["lat"], n["lon"]) <= 500
        )

        # Surface parcs dans 500 m
        features.park_area_500m_m2 = sum(
            _estimate_area_m2(e)
            for e in park_elements
            if _haversine(lat, lon, *_center_of(e)) <= 500
        )


# ------------------------------------------------------------------ #
#  Score de walkabilité                                                #
# ------------------------------------------------------------------ #

def _compute_walkability(f: OSMFeatures) -> float:
    """
    Score composite 0–100.
      Transport (40 pts) : max à 0 m, zéro à 1 000 m
      Parc      (20 pts) : max à 0 m, zéro à 800 m
      École     (20 pts) : max à 0 m, zéro à 600 m
      Commerce  (20 pts) : max à 0 m, zéro à 500 m
    """
    def _s(dist: float | None, max_d: float, w: float) -> float:
        return w * max(0.0, 1.0 - dist / max_d) if dist is not None else 0.0

    return round(
        _s(f.dist_metro_m, 1_000, 40)
        + _s(f.dist_park_m, 800, 20)
        + _s(f.dist_school_m, 600, 20)
        + _s(f.dist_supermarket_m, 500, 20),
        2,
    )


# ------------------------------------------------------------------ #
#  Géométrie (sans dépendance lourde)                                 #
# ------------------------------------------------------------------ #

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _nearest_distance(
    lat: float, lon: float, nodes: list[dict[str, Any]]
) -> float | None:
    dists = [
        _haversine(lat, lon, n["lat"], n["lon"])
        for n in nodes
        if "lat" in n and "lon" in n
    ]
    return round(min(dists), 1) if dists else None


def _extract_centers(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for e in elements:
        if "lat" in e:
            result.append(e)
        elif "center" in e:
            result.append({"lat": e["center"]["lat"], "lon": e["center"]["lon"]})
    return result


def _center_of(element: dict[str, Any]) -> tuple[float, float]:
    if "lat" in element:
        return element["lat"], element["lon"]
    c = element.get("center", {})
    return c.get("lat", 0.0), c.get("lon", 0.0)


def _estimate_area_m2(element: dict[str, Any]) -> float:
    b = element.get("bounds")
    if not b:
        return 0.0
    dlat = _haversine(b["minlat"], b["minlon"], b["maxlat"], b["minlon"])
    dlon = _haversine(b["minlat"], b["minlon"], b["minlat"], b["maxlon"])
    return round(dlat * dlon, 1)
