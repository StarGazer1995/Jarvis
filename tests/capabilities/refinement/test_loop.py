import pytest
from unittest.mock import AsyncMock, patch, mock_open
from src.capabilities.refinement.loop import RefinementLoop


class TestRefinementLoop:
    @pytest.fixture
    def loop(self):
        generator = AsyncMock(return_value="Generated Content")
        reviewer = AsyncMock(return_value="PASS")
        return RefinementLoop(generator=generator, reviewer=reviewer, max_retries=2)

    @pytest.mark.asyncio
    async def test_run_success_first_try(self, loop):
        """Test successful execution on the first attempt."""
        result = await loop.run("Initial Prompt")

        assert result == "Generated Content"
        loop.generator.assert_called_once_with("Initial Prompt")
        loop.reviewer.assert_called_once_with("Generated Content")

    @pytest.mark.asyncio
    async def test_run_refinement_success(self, loop):
        """Test successful execution after one refinement."""
        # First attempt fails, second succeeds
        loop.reviewer.side_effect = ["RETRY: Improve clarity", "PASS"]
        loop.generator.side_effect = ["Bad Content", "Good Content"]

        result = await loop.run("Initial Prompt")

        assert result == "Good Content"
        assert loop.generator.call_count == 2
        assert loop.reviewer.call_count == 2

    @pytest.mark.asyncio
    async def test_run_max_retries_reached(self, loop):
        """Test reaching max retries."""
        loop.reviewer.return_value = "RETRY: Still bad"
        loop.generator.return_value = "Content"

        result = await loop.run("Initial Prompt")

        assert result == "Content"
        assert loop.reviewer.call_count == 2
        assert loop.generator.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_ambiguous_feedback(self, loop):
        """Test handling of ambiguous feedback."""
        loop.reviewer.return_value = "Neither PASS nor RETRY"
        result = await loop.run("Prompt")

        assert result == "Generated Content"
        assert loop.reviewer.call_count == 1
        # Loop breaks immediately

    def test_save_result(self, loop):
        """Test saving results to file."""
        m_open = mock_open()
        with (
            patch("builtins.open", m_open),
            patch("os.makedirs") as mock_makedirs,
            patch("os.getcwd", return_value="/tmp"),
        ):
            path = loop.save_result("Content", prefix="test")

            assert "test_" in path
            assert path.endswith(".md")
            mock_makedirs.assert_called_once()
            m_open.assert_called_once()
            m_open().write.assert_called()

    def test_save_result_with_task(self, loop):
        """Test saving result with task description."""
        m_open = mock_open()
        with (
            patch("builtins.open", m_open),
            patch("os.makedirs"),
            patch("os.getcwd", return_value="/tmp"),
        ):
            loop.save_result("Content", task="My Task")

            args, _ = m_open().write.call_args
            content_written = args[0]
            assert "# Research Report" in content_written
            assert "My Task" in content_written
            assert "Content" in content_written
