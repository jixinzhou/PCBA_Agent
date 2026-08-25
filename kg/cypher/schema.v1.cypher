// T11.1 static Neo4j schema. T11.2 executes and validates this file.
// Every ontology entity receives :KnowledgeEntity plus its subtype label.

CREATE CONSTRAINT knowledge_entity_id_unique IF NOT EXISTS
FOR (node:KnowledgeEntity)
REQUIRE node.entity_id IS UNIQUE;

CREATE CONSTRAINT causal_hypothesis_relationship_id_unique IF NOT EXISTS
FOR (hypothesis:CausalHypothesis)
REQUIRE hypothesis.relationship_id IS UNIQUE;

CREATE INDEX knowledge_entity_canonical_name IF NOT EXISTS
FOR (node:KnowledgeEntity)
ON (node.canonical_name);

CREATE INDEX causal_hypothesis_defect_id IF NOT EXISTS
FOR (hypothesis:CausalHypothesis)
ON (hypothesis.defect_id);

CREATE INDEX causal_hypothesis_strength IF NOT EXISTS
FOR (hypothesis:CausalHypothesis)
ON (hypothesis.relation_strength);

CREATE INDEX causal_hypothesis_verification_status IF NOT EXISTS
FOR (hypothesis:CausalHypothesis)
ON (hypothesis.verification_status);

// Allowed topology for the deterministic importer:
// (:Defect)-[:HAS_HYPOTHESIS]->(:CausalHypothesis)
// (:CausalHypothesis)-[:PROPOSES_CAUSE]->(:CandidateCause)
// (:CandidateCause)-[:BELONGS_TO]->(:Process)
// (:QualityMetric)-[:BELONGS_TO]->(:Process)
// (:Tool)-[:BELONGS_TO]->(:Process)
// (:CausalHypothesis)-[:REQUIRES_METRIC]->(:QualityMetric)
// (:CausalHypothesis)-[:VALIDATED_BY]->(:Tool)
// (:CausalHypothesis)-[:OPTIMIZED_BY]->(:Tool)
