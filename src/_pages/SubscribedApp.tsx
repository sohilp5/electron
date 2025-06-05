// src/_pages/SubscribedApp.tsx
import React, { useEffect, useRef, useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import Queue from "../_pages/Queue";
import Solutions from "../_pages/Solutions";
import Debug from "./Debug"; // Assuming Debug component exists
import GeneralProblemSolver from "./GeneralProblemSolver"; // Import the new component
import { useToast } from "../contexts/toast";

interface SubscribedAppProps {
  credits: number;
  currentLanguage: string;
  setLanguage: (language: string) => void;
}

const SubscribedApp: React.FC<SubscribedAppProps> = ({
  credits,
  currentLanguage,
  setLanguage
}) => {
  const queryClient = useQueryClient();
  const [view, setView] = useState<"queue" | "solutions" | "debug" | "generalProblem">("queue");
  const containerRef = useRef<HTMLDivElement>(null);
  const { showToast } = useToast();

  // Effect for event listeners
  useEffect(() => {
    const cleanupFunctions = [
      window.electronAPI.onResetView(() => {
        queryClient.invalidateQueries({ queryKey: ["screenshots"] });
        queryClient.invalidateQueries({ queryKey: ["problem_statement"] });
        queryClient.invalidateQueries({ queryKey: ["solution"] });
        queryClient.invalidateQueries({ queryKey: ["new_solution"] });
        queryClient.invalidateQueries({ queryKey: ["general_problem_solution"] }); // Clear general solution too
        setView("queue");
      }),
      window.electronAPI.onSolutionStart(() => {
        setView("solutions");
      }),
      window.electronAPI.onUnauthorized(() => {
        // Clear relevant queries and revert to queue
        queryClient.removeQueries({ queryKey: ["screenshots"] });
        queryClient.removeQueries({ queryKey: ["solution"] });
        queryClient.removeQueries({ queryKey: ["problem_statement"] });
        queryClient.removeQueries({ queryKey: ["new_solution"] });
        queryClient.removeQueries({ queryKey: ["general_problem_solution"] });
        setView("queue");
      }),
      window.electronAPI.onProblemExtracted((data: any) => {
        // This is for the coding problem flow
        if (view === "queue" || view === "solutions") { // Ensure it's relevant to current flow
          queryClient.setQueryData(["problem_statement"], data);
        }
      }),
      window.electronAPI.onSolutionError((error: string) => {
        // This is for the coding problem flow
        showToast("Coding Solution Error", error, "error");
         if (view === "solutions") setView("queue"); // Revert if in solutions view
      }),
      window.electronAPI.onSolutionSuccess((data: any) => {
        // This is for the coding problem flow
        if (view === "solutions") { // Only if currently expecting a coding solution
            queryClient.setQueryData(["solution"], data);
        }
      }),

      // --- START OF ADDED/MODIFIED CODE for General Problem Solver ---
      window.electronAPI.onGeneralProblemStart(() => {
        console.log("General Problem Start event received in SubscribedApp");
        queryClient.setQueryData(["general_problem_solution"], null); // Clear previous general solution
        setView("generalProblem");
      }),
      window.electronAPI.onGeneralProblemSuccess((data: any) => {
        console.log("General Problem Success event received in SubscribedApp", data);
        queryClient.setQueryData(["general_problem_solution"], data);
        // The GeneralProblemSolver component will use this query data
        if (view !== "generalProblem") { // Switch view if not already there
            setView("generalProblem");
        }
      }),
      window.electronAPI.onGeneralProblemError((error: string) => {
        console.error("General Problem Error event received in SubscribedApp", error);
        showToast("General Problem Error", error, "error");
        setView("queue"); // Revert to queue on error
      }),
      // --- END OF ADDED/MODIFIED CODE for General Problem Solver ---
    ];

    return () => cleanupFunctions.forEach((fn) => fn());
  }, [queryClient, showToast, view, setView]); // Added view to dependency array

  // Dynamically update the window size
  useEffect(() => {
    if (!containerRef.current) return;

    const updateDimensions = () => {
      if (!containerRef.current) return;
      const height = containerRef.current.scrollHeight || 600;
      const width = containerRef.current.scrollWidth || 800;
      window.electronAPI?.updateContentDimensions({ width, height });
    };

    updateDimensions(); // Initial call
    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(containerRef.current);

    const mutationObserver = new MutationObserver(updateDimensions);
    mutationObserver.observe(containerRef.current, {
      childList: true,
      subtree: true,
      attributes: true,
      characterData: true
    });
    
    const delayedUpdate = setTimeout(updateDimensions, 300); // Shorter delay

    return () => {
      resizeObserver.disconnect();
      mutationObserver.disconnect();
      clearTimeout(delayedUpdate);
    };
  }, [view]); // Re-run when view changes to ensure correct sizing

  return (
    <div ref={containerRef} className="min-h-0">
      {view === "queue" ? (
        <Queue
          setView={setView}
          credits={credits}
          currentLanguage={currentLanguage}
          setLanguage={setLanguage}
        />
      ) : view === "solutions" ? (
        <Solutions
          setView={setView}
          credits={credits}
          currentLanguage={currentLanguage}
          setLanguage={setLanguage}
        />
      ) : view === "debug" ? (
         <Debug 
            isProcessing={false} // This state should be managed within Debug or passed down
            setIsProcessing={() => {}} // This state should be managed within Debug or passed down
            currentLanguage={currentLanguage} 
            setLanguage={setLanguage}
        />
      ) : view === "generalProblem" ? (
        <GeneralProblemSolver
          // currentLanguage might not be strictly necessary here, but pass for consistency
          currentLanguage={currentLanguage}
          setLanguage={setLanguage}
          setView={setView}
        />
      ) : null}
    </div>
  );
};

export default SubscribedApp;
