import React, { useState, useEffect, useRef } from "react"
import { useQuery } from "@tanstack/react-query"
import ScreenshotQueue from "../components/Queue/ScreenshotQueue"
import QueueCommands from "../components/Queue/QueueCommands"
import { useToast } from "../contexts/toast"
import { Screenshot } from "../types/screenshots"

async function fetchScreenshots(): Promise<Screenshot[]> {
  try {
    const existing = await window.electronAPI.getScreenshots()
    return existing
  } catch (error) {
    console.error("Error loading screenshots:", error)
    throw error
  }
}

// Updated interface to include the 'generalProblem' view
interface QueueProps {
  setView: (view: "queue" | "solutions" | "debug" | "generalProblem") => void
  credits: number
  currentLanguage: string
  setLanguage: (language: string) => void
}

const Queue: React.FC<QueueProps> = ({
  setView,
  credits,
  currentLanguage,
  setLanguage
}) => {
  const { showToast } = useToast()
  const [isTooltipVisible, setIsTooltipVisible] = useState(false)
  const [tooltipHeight, setTooltipHeight] = useState(0)
  const contentRef = useRef<HTMLDivElement>(null)

  const {
    data: screenshots = [],
    isLoading,
    refetch
  } = useQuery<Screenshot[]>({
    queryKey: ["screenshots"],
    queryFn: fetchScreenshots,
    staleTime: Infinity,
    gcTime: Infinity,
    refetchOnWindowFocus: false
  })

  const handleDeleteScreenshot = async (index: number) => {
    const screenshotToDelete = screenshots[index]
    if (!screenshotToDelete) return;

    try {
      const response = await window.electronAPI.deleteScreenshot(
        screenshotToDelete.path
      )
      if (response.success) {
        refetch()
      } else {
        console.error("Failed to delete screenshot:", response.error)
        showToast("Error", "Failed to delete the screenshot file", "error")
      }
    } catch (error) {
      console.error("Error deleting screenshot:", error)
    }
  }

  useEffect(() => {
    const updateDimensions = () => {
      if (contentRef.current) {
        let contentHeight = contentRef.current.scrollHeight
        const contentWidth = contentRef.current.scrollWidth
        if (isTooltipVisible) {
          contentHeight += tooltipHeight
        }
        window.electronAPI.updateContentDimensions({
          width: contentWidth,
          height: contentHeight
        })
      }
    }

    const resizeObserver = new ResizeObserver(updateDimensions)
    if (contentRef.current) {
      resizeObserver.observe(contentRef.current)
    }
    updateDimensions()

    const cleanupFunctions = [
      window.electronAPI.onScreenshotTaken(() => refetch()),
      window.electronAPI.onResetView(() => refetch()),
      window.electronAPI.onDeleteLastScreenshot(async () => {
        if (screenshots.length > 0) {
          await handleDeleteScreenshot(screenshots.length - 1);
        } else {
          showToast("No Screenshots", "There are no screenshots to delete", "neutral");
        }
      }),
      window.electronAPI.onSolutionError((error: string) => {
        showToast("Processing Failed", "There was an error processing your screenshots.", "error")
        setView("queue")
        console.error("Processing error:", error)
      }),
      window.electronAPI.onProcessingNoScreenshots(() => {
        showToast("No Screenshots", "There are no screenshots to process.", "neutral")
      }),
    ]

    return () => {
      resizeObserver.disconnect()
      cleanupFunctions.forEach((cleanup) => cleanup())
    }
  }, [isTooltipVisible, tooltipHeight, screenshots, refetch, setView, showToast])

  const handleTooltipVisibilityChange = (visible: boolean, height: number) => {
    setIsTooltipVisible(visible)
    setTooltipHeight(height)
  }
  
  // Logic for the "Solve Code" button
  const handleSolve = async () => {
    if (screenshots.length === 0) {
        showToast("No Screenshots", "Please add at least one screenshot to solve a coding problem.", "error");
        return;
    }
    try {
        await window.electronAPI.triggerProcessScreenshots();
        // The onSolutionStart listener in SubscribedApp will handle switching the view
    } catch (error: any) {
        console.error("Failed to start solving problem:", error);
        showToast("Error", error.message || "Could not start the solution process.", "error");
    }
  };

  // Logic for the "Solve General" button
  const handleSolveGeneral = () => {
    if (screenshots.length === 0) {
      showToast("No Screenshots", "Please add at least one screenshot to start a general problem.", "error");
      return;
    }
    // This now ONLY switches the view. GeneralProblemSolver will handle the rest.
    setView("generalProblem");
  };

  return (
    <div ref={contentRef} className={`bg-transparent w-full flex justify-center p-4`}>
      <div className="space-y-3 w-fit">
        <ScreenshotQueue
          isLoading={isLoading}
          screenshots={screenshots}
          onDeleteScreenshot={handleDeleteScreenshot}
        />
        <QueueCommands
          onTooltipVisibilityChange={handleTooltipVisibilityChange}
          screenshotCount={screenshots.length}
          credits={credits}
          currentLanguage={currentLanguage}
          setLanguage={setLanguage}
          onSolve={handleSolve}
          onSolveGeneral={handleSolveGeneral}
        />
      </div>
    </div>
  )
}

export default Queue