/* CosmicTiger - private PMMM action declarations; BSL-1.0. */
#pragma once

#include "state.hpp"

HPX_DEFINE_PLAIN_ACTION(pmmm::detail::accumulate, PmmmAccumulateAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmAccumulateAction, PmmmAccumulateAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::readCells, PmmmReadCellsAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmReadCellsAction, PmmmReadCellsAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::readOwners, PmmmReadOwnersAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmReadOwnersAction, PmmmReadOwnersAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::readGhostParticles, PmmmReadGhostParticlesAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmReadGhostParticlesAction, PmmmReadGhostParticlesAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::prepare, PmmmPrepareAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmPrepareAction, PmmmPrepareAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::deposit, PmmmDepositAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmDepositAction, PmmmDepositAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::planGhosts, PmmmPlanGhostsAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmPlanGhostsAction, PmmmPlanGhostsAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::startPipeline, PmmmStartPipelineAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmStartPipelineAction, PmmmStartPipelineAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::finishPipeline, PmmmFinishPipelineAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmFinishPipelineAction, PmmmFinishPipelineAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::abortPipeline, PmmmAbortPipelineAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmAbortPipelineAction, PmmmAbortPipelineAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::clearGhosts, PmmmClearGhostsAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmClearGhostsAction, PmmmClearGhostsAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::transform, PmmmTransformAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmTransformAction, PmmmTransformAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::evaluateParticles, PmmmEvaluateAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmEvaluateAction, PmmmEvaluateAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::destroy, PmmmDestroyAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmDestroyAction, PmmmDestroyAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::diagnostics, PmmmDiagnosticsAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmDiagnosticsAction, PmmmDiagnosticsAction);
HPX_DEFINE_PLAIN_ACTION(pmmm::detail::kickResult, PmmmKickResultAction);
HPX_REGISTER_ACTION_DECLARATION(PmmmKickResultAction, PmmmKickResultAction);
