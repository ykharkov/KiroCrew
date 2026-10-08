import { screen, waitFor, within, fireEvent } from '@testing-library/react'
import EmbeddingModelCard from '../pages/overview/EmbeddingModelCard'
import { renderWithProviders } from './helpers'
import { api } from '../api/client'

// A backend that ships a catalog splits the card into two sources: a first-class
// model picked from a dropdown, downloaded from Hugging Face in one click and
// applied in another, or a local GGUF path whose pooling its file declares.

vi.mock('../api/client', () => ({
  api: {
    vectorEmbeddingStatus: vi.fn(),
    vectorValidateEmbedModel: vi.fn(),
    vectorApplyEmbedModel: vi.fn(),
  },
}))

const GEMMA = {
  id: 'embeddinggemma-300m-q8', name: 'EmbeddingGemma-300M', repo: 'ggml-org/embeddinggemma-300M-GGUF',
  size_bytes: 333590944, dim: 768, license: 'Gemma Terms of Use', license_url: 'https://ai.google.dev/gemma/terms',
  downloaded: false, active: false,
}

const BUNDLED = {
  provider: 'llama_cpp', setup_step: 'done', model_available: true,
  model_source: 'default', model_path: '', model_id: 'qwen3-embedding:0.6b', model_dim: 1024, model_active: true,
  reembed: { step: 'idle' }, bundled_model_id: 'qwen3-embedding:0.6b',
  catalog: [GEMMA], catalog_download: { id: '', step: 'idle', error: '' },
}

const GEMMA_ACTIVE = {
  ...BUNDLED, model_source: 'custom', model_path: '/home/models/embeddinggemma-300M-Q8_0.gguf',
  model_id: `${GEMMA.id}:sha256:${'b'.repeat(64)}`, model_dim: 768, catalog: [{ ...GEMMA, downloaded: true, active: true }],
}

function setStatus(status: Record<string, unknown>) {
  vi.mocked(api.vectorEmbeddingStatus).mockResolvedValue(status as never)
  vi.mocked(api.vectorValidateEmbedModel).mockResolvedValue({ ok: true, size_bytes: 2 * 1024 * 1024 } as never)
  vi.mocked(api.vectorApplyEmbedModel).mockResolvedValue({ ok: true } as never)
}

async function pick(name: string) {
  const select = await screen.findByRole('combobox', { name: 'Model' })
  fireEvent.pointerDown(select, { pointerType: 'mouse', button: 0, ctrlKey: false })
  fireEvent.click(await screen.findByRole('option', { name }))
}

function catalogPanel() {
  return within(screen.getByTestId('embed-model-catalog'))
}

describe('EmbeddingModelCard first-class models', () => {
  it('downloads the model picked from the dropdown', async () => {
    setStatus(BUNDLED)
    renderWithProviders(<EmbeddingModelCard />)
    await waitFor(() => expect(catalogPanel().getByText('In use')).toBeInTheDocument())
    await pick('EmbeddingGemma-300M')
    fireEvent.click(catalogPanel().getByRole('button', { name: /Download/ }))
    await waitFor(() => expect(api.vectorApplyEmbedModel).toHaveBeenCalledWith('', { catalog_id: GEMMA.id, download: true }))
  })

  it('applies a downloaded model through the confirm modal', async () => {
    setStatus({ ...BUNDLED, catalog: [{ ...GEMMA, downloaded: true }] })
    renderWithProviders(<EmbeddingModelCard />)
    await pick('EmbeddingGemma-300M')
    fireEvent.click(catalogPanel().getByRole('button', { name: 'Apply model' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Change model' }))
    await waitFor(() => expect(api.vectorApplyEmbedModel).toHaveBeenCalledWith('', { catalog_id: GEMMA.id }))
  })

  it('names the active model and offers the bundled one back', async () => {
    setStatus(GEMMA_ACTIVE)
    renderWithProviders(<EmbeddingModelCard />)
    await waitFor(() => expect(screen.getByTestId('embed-model-active')).toHaveTextContent('EmbeddingGemma-300M'))
    expect(screen.getByRole('combobox', { name: 'Model' })).toHaveTextContent('EmbeddingGemma-300M')
    expect(catalogPanel().getByText('In use')).toBeInTheDocument()
    await pick('Qwen3-Embedding-0.6B')
    fireEvent.click(catalogPanel().getByRole('button', { name: 'Apply model' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Change model' }))
    await waitFor(() => expect(api.vectorApplyEmbedModel).toHaveBeenCalledWith(''))
  })

  it('keeps a local path on its own tab, applied as a path', async () => {
    setStatus(BUNDLED)
    renderWithProviders(<EmbeddingModelCard />)
    fireEvent.click(await screen.findByRole('radio', { name: 'Local file' }))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '/models/other.gguf' } })
    fireEvent.click(within(screen.getByTestId('embed-model-card')).getByRole('button', { name: 'Apply model' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Change model' }))
    await waitFor(() => expect(api.vectorApplyEmbedModel).toHaveBeenCalledWith('/models/other.gguf'))
  })
})
