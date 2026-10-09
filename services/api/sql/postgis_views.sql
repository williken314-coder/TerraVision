-- Spatial views over the project model for GIS tools (QGIS, ArcGIS) when running on PostGIS.
-- Coordinates are project feet: x east, z south, so y (north) = -z. Set the SRID once the
-- project is registered to a real coordinate system (datum.epsg).
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE OR REPLACE VIEW gv_structures AS
SELECT p.id AS project_id,
       s->>'id' AS structure_id,
       s->>'system' AS system,
       s->>'kind' AS kind,
       (s->'rim'->>'value')::float AS rim,
       s->'rim'->>'status' AS rim_status,
       s->'rim'->>'source' AS rim_sheet,
       ST_SetSRID(ST_MakePoint((s->>'x')::float, -(s->>'z')::float), COALESCE((p.model->'datum'->>'epsg')::int, 0)) AS geom
FROM projects p, json_array_elements(p.model::json->'structures') s;

CREATE OR REPLACE VIEW gv_pipes AS
SELECT p.id AS project_id,
       pp->>'id' AS pipe_id,
       pp->>'system' AS system,
       pp->>'from_id' AS from_id,
       pp->>'to_id' AS to_id,
       (pp->'dia_in'->>'value')::float AS dia_in,
       (pp->'inv_up'->>'value')::float AS inv_up,
       (pp->'inv_dn'->>'value')::float AS inv_dn,
       ST_SetSRID(ST_MakeLine(ST_MakePoint((a->>'x')::float, -(a->>'z')::float), ST_MakePoint((b->>'x')::float, -(b->>'z')::float)),
                  COALESCE((p.model->'datum'->>'epsg')::int, 0)) AS geom
FROM projects p,
     json_array_elements(p.model::json->'pipes') pp,
     json_array_elements(p.model::json->'structures') a,
     json_array_elements(p.model::json->'structures') b
WHERE a->>'id' = pp->>'from_id' AND b->>'id' = pp->>'to_id';
