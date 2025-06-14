// src/_pages/GeneralProblemSolver.tsx
import React, { useState, useEffect, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import ScreenshotQueue from "../components/Queue/ScreenshotQueue";
import { useToast } from "../contexts/toast";
import { Screenshot as ScreenshotType } from "../types";
import { Marked } from 'marked';
import 'katex/dist/katex.min.css';
import markedKatex from 'marked-katex-extension';

// Define interfaces for clarity
interface GeneralProblemSolverProps {
    setView: (view: "queue" | "solutions" | "debug" | "generalProblem") => void;
    currentLanguage: string;
    setLanguage: (language: string) => void;
}

interface Analysis {
    screenshotPath: string;
    solution: string;
}

// --- Markdown and KaTeX Setup ---
const katexExtensionOptions = {
  throwOnError: false,
  delimiters: [
    { left: "$$", right: "$$", display: true },
    { left: "$", right: "$", display: false },
    { left: "\\(", right: "\\)", display: false },
    { left: "\\[", right: "\\]", display: true }
  ],
};
const customMarked = new Marked();
customMarked.use(markedKatex(katexExtensionOptions));
// --- End Setup ---


const GeneralProblemSolver: React.FC<GeneralProblemSolverProps> = ({ setView }) => {
    const { showToast } = useToast();
    const contentRef = useRef<HTMLDivElement>(null);

    // State Management
    const [analyses, setAnalyses] = useState<Analysis[]>([]);
    const [summary, setSummary] = useState<string | null>(null);
    const [activeView, setActiveView] = useState<number | 'summary'>(0);
    const [isProcessing, setIsProcessing] = useState(false);

    // Fetch available screenshots using react-query
    const { data: screenshots = [], refetch: refetchScreenshots } = useQuery<ScreenshotType[]>({
        queryKey: ["general_problem_screenshots"],
        queryFn: async () => {
            // The onScreenshotTaken listener in SubscribedApp invalidates this query, causing a refetch.
            const response = await window.electronAPI.getScreenshots();
            const previews = Array.isArray(response) ? response : response?.previews || [];
            return previews.map((p: { path: string; preview: string; }) => ({
                id: p.path,
                path: p.path,
                timestamp: Date.now(),
                thumbnail: p.preview,
            }));
        },
    });

    // When a screenshot is taken anywhere, SubscribedApp refetches the query, and this listener updates our local state.
    useEffect(() => {
        const cleanup = window.electronAPI.onScreenshotTaken(() => {
            showToast("Screenshot Added", "New screenshot has been added to the case.", "success");
            refetchScreenshots();
        });
        return cleanup;
    }, [refetchScreenshots, showToast]);


    const processedPaths = analyses.map(a => a.screenshotPath);
    const unprocessedScreenshots = screenshots.filter(s => !processedPaths.includes(s.path));

    // --- Core Logic Handlers ---

    const handleAddScreenshot = async () => {
        try {
            await window.electronAPI.triggerScreenshot();
            // The onScreenshotTaken listener will handle the success toast and refetch.
        } catch (error: any) {
            showToast("Error", "Could not take a screenshot.", "error");
            console.error(error);
        }
    };

    const handleProcessNext = useCallback(async () => {
        if (unprocessedScreenshots.length === 0) {
            showToast("No More Screenshots", "All available screenshots have been analyzed.", "neutral");
            return;
        }
        setIsProcessing(true);

        const screenshotToProcess = unprocessedScreenshots[0];
        const contextPaths = processedPaths; 

        try {
            const result = await window.electronAPI.processGeneralProblem(screenshotToProcess.path, contextPaths);
            if (result.success && result.data?.solution) {
                const newAnalysis: Analysis = {
                    screenshotPath: screenshotToProcess.path,
                    solution: result.data.solution,
                };
                setAnalyses(prev => [...prev, newAnalysis]);
                setActiveView(analyses.length); 
                showToast("Analysis Complete", `Analysis for screenshot ${analyses.length + 1} is ready.`, "success");
            } else {
                throw new Error(result.error || "Failed to get analysis from the AI.");
            }
        } catch (error: any) {
            showToast("Processing Error", error.message, "error");
        } finally {
            setIsProcessing(false);
        }
    }, [unprocessedScreenshots, processedPaths, analyses.length, showToast]);

    const handleSummarize = async () => {
        if (analyses.length === 0) {
            showToast("Nothing to Summarize", "Please analyze at least one screenshot first.", "error");
            return;
        }
        setIsProcessing(true);
        try {
            const analysisTexts = analyses.map(a => a.solution);
            const result = await window.electronAPI.summarizeGeneralProblem(analysisTexts);
            if (result.success && result.data?.summary) {
                setSummary(result.data.summary);
                setActiveView('summary');
                showToast("Summary Ready", "The final case summary has been generated.", "success");
            } else {
                throw new Error(result.error || "Failed to generate summary.");
            }
        } catch (error: any) {
            showToast("Summarization Error", error.message, "error");
        } finally {
            setIsProcessing(false);
        }
    };

    // --- Component Effects ---

    // Automatically process the first screenshot on mount
    useEffect(() => {
        if (screenshots.length > 0 && analyses.length === 0 && !isProcessing) {
            handleProcessNext();
        }
    }, [screenshots.length, analyses.length, isProcessing, handleProcessNext]);


    // Update window dimensions when content changes
    useEffect(() => {
        if (!contentRef.current) return;
        const resizeObserver = new ResizeObserver(() => {
            if (contentRef.current) {
                const height = contentRef.current.scrollHeight;
                const width = contentRef.current.scrollWidth;
                window.electronAPI?.updateContentDimensions({ width, height });
            }
        });
        resizeObserver.observe(contentRef.current);
        return () => resizeObserver.disconnect();
    }, [analyses, summary, activeView]);

    // --- Render Helpers ---

    const getContentHtml = (): { __html: string } => {
        const content = activeView === 'summary' ? summary : analyses[activeView]?.solution;
        if (!content) return { __html: "" };
        try {
            return { __html: customMarked.parse(content, { breaks: true, gfm: true }) as string };
        } catch (e) {
            return { __html: `<pre>${content.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</pre>` };
        }
    };

    return (
        <div ref={contentRef} className="relative p-4 space-y-4 bg-black/80 text-white rounded-lg w-full">
            <div className="flex justify-between items-center mb-4">
                <h1 className="text-2xl font-semibold text-white/90">Case Interview Strategist</h1>
                <button onClick={() => setView("queue")} className="bg-gray-700 hover:bg-gray-600 text-white/90 font-medium py-2 px-4 rounded text-xs transition-colors">
                    Back to Queue
                </button>
            </div>

            <div className="mb-6">
                <h2 className="text-lg font-medium text-white/80 mb-2">Case Materials:</h2>
                <ScreenshotQueue
                    screenshots={screenshots.map(s => ({ path: s.path, preview: s.thumbnail }))}
                    isLoading={false}
                    onDeleteScreenshot={() => { /* Deleting mid-case is disabled to maintain analysis order */ }}
                />
            </div>
            
            <div className="grid grid-cols-3 items-center gap-2 p-2 bg-black/50 border border-white/10 rounded-md">
                {/* NEW BUTTON to add a screenshot */}
                <button onClick={handleAddScreenshot} disabled={isProcessing} className="bg-gray-600 hover:bg-gray-500 text-white font-medium py-2 px-4 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                    Add Screenshot
                </button>

                <button onClick={handleProcessNext} disabled={isProcessing || unprocessedScreenshots.length === 0} className="bg-green-600 hover:bg-green-500 text-white font-medium py-2 px-4 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                    {isProcessing ? "Processing..." : `Process Next (${unprocessedScreenshots.length})`}
                </button>
                <button onClick={handleSummarize} disabled={isProcessing || analyses.length === 0} className="bg-blue-600 hover:bg-blue-500 text-white font-medium py-2 px-4 rounded text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                    {isProcessing ? "..." : "Summarize"}
                </button>
            </div>

            <div className="flex flex-wrap items-center gap-2 mt-4 p-2 bg-black/30 rounded-md">
                <span className="text-sm font-semibold mr-2 text-white/70">View:</span>
                {analyses.map((_, index) => (
                    <button key={`view-${index}`} onClick={() => setActiveView(index)} disabled={activeView === index} className={`px-3 py-1 text-xs rounded transition-colors ${activeView === index ? 'bg-white text-black font-semibold' : 'bg-white/10 hover:bg-white/20 text-white'}`}>
                        {`Analysis ${index + 1}`}
                    </button>
                ))}
                {summary && (
                    <button onClick={() => setActiveView('summary')} disabled={activeView === 'summary'} className={`px-3 py-1 text-xs rounded transition-colors ${activeView === 'summary' ? 'bg-blue-500 text-white font-semibold' : 'bg-white/10 hover:bg-white/20 text-white'}`}>
                        Final Summary
                    </button>
                )}
            </div>

            <div className="mt-4 min-h-[200px]">
                {isProcessing && (
                     <div className="flex flex-col items-center justify-center py-10">
                        <div className="w-8 h-8 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                        <p className="text-md text-white/70 mt-3">AI Strategist is thinking...</p>
                    </div>
                )}
                {!isProcessing && analyses.length === 0 && screenshots.length > 0 && (
                     <div className="text-center text-gray-400 py-10">
                        <p>Ready to begin analysis.</p>
                        <p className="text-xs mt-1">Click "Process Next" to analyze the first screenshot.</p>
                    </div>
                )}
                <div
                    className="prose prose-sm prose-invert max-w-none bg-black/60 border border-gray-700/50 p-4 rounded-md"
                    dangerouslySetInnerHTML={getContentHtml()}
                />
            </div>
        </div>
    );
};

export default GeneralProblemSolver;